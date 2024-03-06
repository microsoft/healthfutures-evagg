import logging
import urllib.parse as urlparse
from typing import Any, Dict, List, Optional, Sequence

from pydantic import Extra, root_validator

from lib.config import PydanticYamlModel
from lib.evagg.svc import IWebContentClient
from lib.evagg.types import Paper

from .interfaces import IAnnotateEntities, IGeneLookupClient, IPaperLookupClient, IVariantLookupClient

logger = logging.getLogger(__name__)


class NcbiApiSettings(PydanticYamlModel, extra=Extra.forbid):
    api_key: Optional[str] = None
    email: str = "biomedcomp@microsoft.com"

    def get_key_string(self) -> Optional[str]:
        key_string = ""
        if self.email:
            key_string += f"&email={urlparse.quote(self.email)}"
        if self.api_key:
            key_string += f"&api_key={self.api_key}"
        return key_string if key_string else None

    @root_validator(pre=True)
    @classmethod
    def _validate_settings(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("api_key") and not values.get("email"):
            raise ValueError("If NCBI_EUTILS_API_KEY is specified NCBI_EUTILS_EMAIL is required.")
        return values


class NcbiLookupClient(IPaperLookupClient, IGeneLookupClient, IVariantLookupClient, IAnnotateEntities):
    """A client for querying the various services in the NCBI API."""

    # According to https://support.nlm.nih.gov/knowledgebase/article/KA-05316/en-us the max
    # RPS for NCBI API endpoints is 3 without an API key, and 10 with an API key.
    SYMBOL_GET_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/gene/symbol/{symbols}/taxon/Human"
    PMCOA_GET_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
    PUBTATOR_GET_URL = (
        "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/pmc_export/bioc{fmt}?pmcids={id}"
    )
    # TODO: consider unicode encoding for the BioC response.
    BIOC_GET_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{pmcid}/ascii"

    EUTILS_HOST = "https://eutils.ncbi.nlm.nih.gov"
    EUTILS_FETCH_URL = "/entrez/eutils/efetch.fcgi?db={db}&id={id}&retmode={retmode}&rettype={rettype}&tool=biopython"
    EUTILS_SEARCH_URL = "/entrez/eutils/esearch.fcgi?db={db}&term={term}&sort={sort}&retmax={retmax}&tool=biopython"

    def __init__(self, web_client: IWebContentClient, settings: Optional[Dict[str, str]] = None) -> None:
        self._config = NcbiApiSettings(**settings) if settings else NcbiApiSettings()
        self._default_max_papers = 5  # TODO: make configurable?
        self._web_client = web_client

    def _esearch(self, db: str, term: str, sort: str, retmax: int, retmode: str | None = None) -> Any:
        key_string = self._config.get_key_string()
        url = self.EUTILS_SEARCH_URL.format(db=db, term=term, sort=sort, retmax=retmax)
        return self._web_client.get(f"{self.EUTILS_HOST}{url}", content_type=retmode, url_extra=key_string)

    def _efetch(self, db: str, id: str, retmode: str | None = None, rettype: str | None = None) -> Any:
        key_string = self._config.get_key_string()
        url = self.EUTILS_FETCH_URL.format(db=db, id=id, retmode=retmode, rettype=rettype)
        return self._web_client.get(f"{self.EUTILS_HOST}{url}", content_type=retmode, url_extra=key_string)

    def _get_oa_props(self, pmcid: str) -> Dict[str, Any]:
        """Get the OA status for a paper from the PMC OA API."""
        props = {"is_pmc_oa": False, "license": "unknown"}
        if not pmcid:
            return props

        # Do a record lookup for the given pmcid at the PMC OA endpoint.
        root = self._web_client.get(self.PMCOA_GET_URL.format(pmcid=pmcid), content_type="xml")
        # Look for a record with the given pmcid in the response.
        record = root.find(f"records/record[@id='{pmcid}']")

        if record is None:
            # No valid OA record returned - if there is an error code, extract it from the response.
            err_code = root.find("error").attrib["code"] if root.find("error") is not None else None
            if err_code == "idIsNotOpenAccess":
                props["license"] = "not_open_access"
            elif err_code:
                logger.warning(f"Unexpected PMC OA error code: {err_code}")
        else:
            props["is_pmc_oa"] = True
            props["license"] = license = record.attrib.get("license", "unknown")
            logger.debug(f"PMC OA record found for {pmcid}")
            if "-ND" in license:
                # TODO if it has a "no derivatives" license, then we don't consider it open access.
                logger.warning(f"PMC OA record found for {pmcid} but has a no-derivatives license: {license}")
                props["is_pmc_oa"] = False

        return props

    def _get_full_text_xml(self, pmcid: str | None, is_pmc_oa: bool, license: str) -> Optional[str]:
        """Get the full text of a paper from PMC."""
        if not pmcid or not is_pmc_oa or license.find("nd") >= 0:
            logger.warning(f"Cannot fetch full text, paper 'pmcid:{pmcid}' is not in PMC-OA or has unusable license.")
            return None

        response_root = self._web_client.get(self.BIOC_GET_URL.format(pmcid=pmcid), content_type="xml")

        # Find and return the specific document.
        for document in response_root.findall("./document"):
            id = document.find("id")
            if id is not None and id.text == pmcid.upper().lstrip("PMC"):
                return document
        logger.warning(f"Response received from BioC, but corresponding PMC ID not found: {pmcid}")
        return None

    def _full_text_sections_from_xml(self, root: Any) -> Sequence[str]:
        """Extract the full text from a BioC XML response."""
        return [p.text for p in root.findall("./passage/text")]

    # IPaperLookupClient

    def search(self, query: str, max_papers: Optional[int] = None) -> Sequence[str]:
        retmax = max_papers or self._default_max_papers
        root = self._esearch(db="pubmed", term=query, sort="relevance", retmax=retmax, retmode="xml")
        pmids = [id.text for id in root.findall("./IdList/Id") if id.text]
        return pmids

    def fetch(self, paper_id: str) -> Optional[Paper]:
        if (root := self._efetch(db="pubmed", id=paper_id, retmode="xml", rettype="abstract")) is None:
            return None

        if (article := root.find(f"PubmedArticle/MedlineCitation/PMID[.='{paper_id}']/../..")) is None:
            return None

        props = _extract_paper_props_from_xml(article)
        props.update(self._get_oa_props(props["pmcid"]))
        props["citation"] = f"{props['first_author']} ({props['pub_year']}) {props['journal']}, {props['doi']}"
        props["pmid"] = paper_id
        props["full_text_xml"] = self._get_full_text_xml(
            pmcid=props["pmcid"], is_pmc_oa=props["is_pmc_oa"], license=props["license"]
        )
        props["full_text_sections"] = (
            self._full_text_sections_from_xml(props["full_text_xml"]) if props["full_text_xml"] is not None else []
        )

        return Paper(id=props["doi"], **props)

    # IGeneLookupClient
    def gene_id_for_symbol(self, *symbols: str, allow_synonyms: bool = False) -> Dict[str, int]:
        """Query the NCBI gene database for the gene_id for a given collection of `symbols`.

        If `allow_synonyms` is True, then this will attempt to return the most relevant gene_id for each symbol. If
        there are multiple matches to a symbol, the direct match (where the query symbol is the official symbol) will
        be returned. If there are no direct matches, then the first synonym match will be returned.
        """
        url = self.SYMBOL_GET_URL.format(symbols=",".join(symbols))
        root = self._web_client.get(url, content_type="json")
        return _extract_gene_symbols(root.get("reports", []), symbols, allow_synonyms)

    # IVariantLookupClient
    def hgvs_from_rsid(self, *rsids: str) -> Dict[str, Dict[str, str]]:
        # Provided rsids should be numeric strings prefixed with `rs`.
        if not rsids or not all(rsid.startswith("rs") and rsid[2:].isnumeric() for rsid in rsids):
            raise ValueError("Invalid rsids list - must provide 'rs' followed by a string of numeric characters.")

        uids = {rsid[2:] for rsid in rsids}
        root = self._efetch(db="snp", id=",".join(uids), retmode="xml", rettype="xml")
        return {"rs" + uid: _extract_hgvs_from_xml(root, uid) for uid in uids}

    # IAnnotateEntities
    def annotate(self, paper: Paper) -> Dict[str, Any]:
        """Annotate the paper with entities from PubTator."""
        if not paper.props.get("pmcid") or not paper.props.get("is_pmc_oa"):
            logger.warning(f"Cannot annotate, paper '{paper}' is not in PMC-OA.")
            return {}

        url = self.PUBTATOR_GET_URL.format(fmt="json", id=paper.props["pmcid"])
        return self._web_client.get(url, content_type="json")


def _extract_hgvs_from_xml(root: Any, uid: str) -> Dict[str, str]:
    if root is None:
        return {}
    ns = "{https://www.ncbi.nlm.nih.gov/SNP/docsum}"
    # Find the first DOCSUM node under a DocumentSummary with the given rsid in the document hierarchy.
    node = next(iter(root.findall(f"./{ns}DocumentSummary[@uid='{uid}']/{ns}DOCSUM")), None)
    if node is None or not node.text:
        return {}

    # Extract all key/value pairs from node text of the form 'key=value|key=value|...' into a dict.
    props = {k: v for k, v in (kvp.split("=") for kvp in (node.text or "").split("|") if "=" in kvp) if k and v}
    # Extract all values from the HGVS property of the form 'HGVS=value1,value2...'.
    hgvs = props.get("HGVS", "").split(",")
    # Return a dict with the first occurrence of each value that starts with 'NP_' (hgvs_p) or 'NM_' (hgvs_c).
    types = {"hgvs_p": lambda x: x.startswith("NP_"), "hgvs_c": lambda x: x.startswith("NM_")}
    return {k: next(filter(match, hgvs)) for k, match in types.items() if (any(map(match, hgvs)))}


def _extract_gene_symbols(reports: List[Dict], symbols: Sequence[str], allow_synonyms: bool) -> Dict[str, int]:
    matches = {g["gene"]["symbol"]: int(g["gene"]["gene_id"]) for g in reports if g["gene"]["symbol"] in symbols}

    if allow_synonyms:
        for missing_symbol in [s for s in symbols if s not in matches.keys()]:
            if synonym := next((g["gene"] for g in reports if missing_symbol in g["query"]), None):
                matches[missing_symbol] = int(synonym["gene_id"])

    return matches


def _extract_paper_props_from_xml(article: Any) -> Dict[str, Any]:
    """Extracts paper properties from an XML root element."""
    extractions = {
        "title": "./MedlineCitation/Article/ArticleTitle",
        "abstract": "./MedlineCitation/Article/Abstract/AbstractText",
        "journal": "./MedlineCitation/Article/Journal/ISOAbbreviation",
        "first_author": "./MedlineCitation/Article/AuthorList/Author[1]/LastName",
        "pub_year": "./MedlineCitation/Article/Journal/JournalIssue/PubDate/Year",
        "doi": "./PubmedData/ArticleIdList/ArticleId[@IdType='doi']",
        "pmcid": "./PubmedData/ArticleIdList/ArticleId[@IdType='pmc']",
    }

    props = {
        k: ("".join(v.itertext()) if (v := article.find(path)) is not None else None) for k, path in extractions.items()
    }
    return props
