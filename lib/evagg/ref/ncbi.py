import logging
import urllib.parse as urlparse
from abc import ABC
from typing import Any, Dict, List, Optional, Sequence

from defusedxml import ElementTree
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


class NcbiClientBase(ABC):
    EUTILS_HOST = "https://eutils.ncbi.nlm.nih.gov"
    EUTILS_FETCH_URL = "/entrez/eutils/efetch.fcgi?db={db}&id={id}&retmode={retmode}&rettype={rettype}&tool=biopython"
    EUTILS_SEARCH_URL = "/entrez/eutils/esearch.fcgi?db={db}&term={term}&sort={sort}&tool=biopython"

    def __init__(self, web_client: IWebContentClient, settings: Optional[Dict[str, str]] = None) -> None:
        self._config = NcbiApiSettings(**settings) if settings else NcbiApiSettings()
        self._web_client = web_client

    def _esearch(self, db: str, term: str, sort: str, **extra_params: Dict[str, Any]) -> Any:
        key_string = self._config.get_key_string()
        url = self.EUTILS_SEARCH_URL.format(db=db, term=term, sort=sort)
        url += "".join([f"&{k}={v}" for k, v in extra_params.items()])
        return self._web_client.get(f"{self.EUTILS_HOST}{url}", content_type="xml", url_extra=key_string)

    def _efetch(self, db: str, id: str, retmode: str | None = None, rettype: str | None = None) -> Any:
        key_string = self._config.get_key_string()
        url = self.EUTILS_FETCH_URL.format(db=db, id=id, retmode=retmode, rettype=rettype)
        return self._web_client.get(f"{self.EUTILS_HOST}{url}", content_type=retmode, url_extra=key_string)


PAPER_BASE_PROPS = {
    "id",
    "pmid",
    "title",
    "abstract",
    "journal",
    "first_author",
    "pub_year",
    "doi",
    "pmcid",
    "citation",
    "can_access",
    "license",
    "link",
}
PAPER_FULL_TEXT_PROPS = {
    "fulltext_xml",
}


class NcbiLookupClient(NcbiClientBase, IPaperLookupClient, IGeneLookupClient, IVariantLookupClient, IAnnotateEntities):
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

    def __init__(self, web_client: IWebContentClient, settings: Optional[Dict[str, str]] = None) -> None:
        super().__init__(web_client, settings)

    def _get_xml_props(self, article: Any) -> Dict[str, str]:
        """Extracts paper properties from an XML root element."""
        extractions = {
            "title": "./MedlineCitation/Article/ArticleTitle",
            "abstract": "./MedlineCitation/Article/Abstract",
            "journal": "./MedlineCitation/Article/Journal/ISOAbbreviation",
            "first_author": "./MedlineCitation/Article/AuthorList/Author[1]/LastName",
            "pub_year": "./MedlineCitation/Article/Journal/JournalIssue/PubDate/Year",
            "doi": "./PubmedData/ArticleIdList/ArticleId[@IdType='doi']",
            "pmcid": "./PubmedData/ArticleIdList/ArticleId[@IdType='pmc']",
        }

        def _get_xml_string(node: Any) -> str:
            return " ".join(("".join(node.itertext())).split()) if node is not None else ""

        props = {k: _get_xml_string(article.find(path)) for k, path in extractions.items()}
        props["citation"] = f"{props['first_author']} ({props['pub_year']}) {props['journal']}, {props['doi']}"
        props["abstract"] = self._join_abstract_sections(article, extractions["abstract"])
        return props

    def _get_license_props(self, pmcid: str) -> Dict[str, str | bool]:
        """Get the access status for a paper from the PMC OA API."""
        props: Dict[str, str | bool] = {"can_access": False, "license": "unknown"}
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
            props["can_access"] = True
            props["license"] = license = record.attrib.get("license", "unknown")
            if "-ND" in license:
                # TODO if it has a "no derivatives" license, then we don't consider it open access.
                logger.debug(f"PMC OA record found for {pmcid} but has a no-derivatives license: {license}")
                props["can_access"] = False

        return props

    def _get_derived_props(self, props: Dict[str, Any]) -> Dict[str, str]:
        """Get the derived properties of a paper."""
        derived_props: Dict[str, str] = {}
        derived_props["citation"] = f"{props['first_author']} ({props['pub_year']}) {props['journal']}, {props['doi']}"
        derived_props["link"] = f"https://pubmed.ncbi.nlm.nih.gov/{props['pmid']}/"
        return derived_props

    def _get_full_text(self, props: Dict[str, Any]) -> str:
        """Get the full text of a paper from PMC."""
        pmcid = props["pmcid"]
        if not props["can_access"]:
            logger.debug(f"Cannot fetch full text, paper 'pmcid:{pmcid}' is not in PMC-OA or has unusable license.")
            return ""
        try:
            root = self._web_client.get(self.BIOC_GET_URL.format(pmcid=pmcid), content_type="xml")
        except Exception as e:
            logger.warning(f"Unexpected error fetching BioC entry for {pmcid}: {e}")
            return ""

        # Find and return the specific document.
        if (doc := root.find(f"./document[id='{pmcid.upper().lstrip('PMC')}']")) is None:
            logger.warning(f"Response received from BioC, but corresponding PMC ID not found: {pmcid}")
        return ElementTree.tostring(doc, encoding="unicode")

    # IPaperLookupClient
    def search(self, query: str, **extra_params: Dict[str, Any]) -> Sequence[str]:
        root = self._esearch(db="pubmed", term=query, sort="relevance", **extra_params)
        pmids = [id.text for id in root.findall("./IdList/Id") if id.text]
        return pmids

    def fetch(self, paper_id: str, include_fulltext: bool = False) -> Optional[Paper]:
        if (root := self._efetch(db="pubmed", id=paper_id, retmode="xml", rettype="abstract")) is None:
            return None

        if (article := root.find(f"PubmedArticle/MedlineCitation/PMID[.='{paper_id}']/../..")) is None:
            return None

        props: Dict[str, Any] = {"id": f"pmid:{paper_id}", "pmid": paper_id}
        props.update(self._get_xml_props(article))
        props.update(self._get_license_props(props["pmcid"]))
        props.update(self._get_derived_props(props))
        assert PAPER_BASE_PROPS == set(props.keys()), f"Missing properties: {PAPER_BASE_PROPS ^ set(props.keys())}"
        if include_fulltext:
            props["fulltext_xml"] = self._get_full_text(props)
        return Paper(**props)

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
        if not paper.props.get("can_access", False):
            logger.warning(f"Cannot annotate, paper '{paper}' is not licensed for access.")
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
