import bisect
import logging
import urllib.parse as urlparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from defusedxml import ElementTree
from pydantic import BaseModel, model_validator
from ratarmountcore.mountsource.factory import open_mount_source
from requests.exceptions import HTTPError, RetryError

from lib.evagg.types import Paper
from lib.evagg.utils import IWebContentClient

from .interfaces import (
    IAnnotateEntities,
    IGeneLookupClient,
    IPaperLookupClient,
    IVariantLookupClient,
)

logger = logging.getLogger(__name__)


class _PmcMirror:
    """Helper class for managing PMC archive mirror."""

    def __init__(self, mirror_dir: Path) -> None:
        self.mirror_dir = mirror_dir
        # Get sorted list of archive prefixes for bisect search
        archives = sorted(mirror_dir.glob("PMC*_xml_ascii.tar.gz"))
        self.archive_prefixes = [int(a.name[3:6]) for a in archives]
        # Cache for opened mount sources
        self._mount_cache: dict[str, Any] = {}
        logger.debug(f"PMC mirror initialized with {len(self.archive_prefixes)} archives")

    def get_archive_path(self, pmcid: str) -> Path | None:
        """Find the archive containing a given PMC ID."""
        # Extract numeric part and get archive prefix by dividing by 100,000
        # E.g., PMC9494810 -> 9494810 // 100000 = 94
        numeric_part = pmcid.upper()[3:]  # Remove "PMC" prefix
        if not numeric_part.isdigit():
            logger.warning(f"Invalid PMC ID format: {pmcid}")
            return None

        # Get archive prefix by dividing by 100,000
        prefix_num = int(numeric_part) // 100000

        # Find the last archive prefix that is <= our PMC prefix
        idx = bisect.bisect_right(self.archive_prefixes, prefix_num) - 1
        if idx < 0:
            logger.debug(f"No archive found for {pmcid}")
            return None

        # Format archive name
        archive_prefix = f"PMC{self.archive_prefixes[idx]:03d}"
        archive_name = f"{archive_prefix}XXXXX_xml_ascii.tar.gz"
        archive_path = self.mirror_dir / archive_name

        if not archive_path.exists():
            logger.debug(f"Archive not found: {archive_path}")
            return None

        return archive_path

    def fetch_xml(self, pmcid: str) -> str:
        """Fetch XML content for a PMC ID from the mirror."""
        archive_path = self.get_archive_path(pmcid)
        if not archive_path:
            raise ValueError(f"No archive found for {pmcid}")

        logger.debug(f"Attempting to fetch {pmcid} from archive: {archive_path.name}")
        xml_filename = f"{pmcid}.xml"

        try:
            # Check if mount source is already cached
            archive_path_str = str(archive_path)
            if archive_path_str not in self._mount_cache:
                # Open the specific archive on-demand. We don't use an AutoMountLayer with
                # an overlaid file system across all archives, as that takes a really long
                # time to initialize.
                logger.debug(f"Opening new mount source for: {archive_path.name}")
                self._mount_cache[archive_path_str] = open_mount_source(archive_path_str)

            mount = self._mount_cache[archive_path_str]
            file_info = mount.lookup(f"/{xml_filename}")
            if not file_info:
                raise ValueError(f"File {xml_filename} not found in archive")

            with mount.open(file_info) as f:
                content = f.read().decode("utf-8")
                logger.info(f"Successfully fetched {pmcid} from local PMC mirror")
                return content
        except Exception as e:
            logger.error(f"Failed to fetch {pmcid} from local PMC mirror: {e}")
            raise ValueError(f"Paper {pmcid} not found in local PMC mirror") from e


class NcbiApiSettings(BaseModel):
    model_config = {"extra": "forbid"}
    api_key: str | None = None
    email: str = "biomedcomp@microsoft.com"

    def get_key_string(self) -> str | None:
        key_string = ""
        if self.email:
            key_string += f"&email={urlparse.quote(self.email)}"
        if self.api_key:
            key_string += f"&api_key={self.api_key}"
        return key_string if key_string else None

    @model_validator(mode="before")
    @classmethod
    def _validate_settings(cls, values: dict[str, Any]) -> dict[str, Any]:
        if values.get("api_key") and not values.get("email"):
            raise ValueError("If NCBI_EUTILS_API_KEY is specified NCBI_EUTILS_EMAIL is required.")
        return values


class NcbiClientBase:
    EUTILS_HOST = "https://eutils.ncbi.nlm.nih.gov"
    EUTILS_SEARCH_SITE = "/entrez/eutils/esearch.fcgi"
    EUTILS_FETCH_SITE = "/entrez/eutils/efetch.fcgi"
    EUTILS_SEARCH_URL = EUTILS_SEARCH_SITE + "?db={db}&term={term}&sort={sort}&tool=biopython"
    EUTILS_FETCH_URL = EUTILS_FETCH_SITE + "?db={db}&id={id}&retmode={retmode}&rettype={rettype}&tool=biopython"

    def __init__(self, web_client: IWebContentClient, settings: dict[str, str] | None = None) -> None:
        self._config = NcbiApiSettings(**settings) if settings else NcbiApiSettings()
        self._web_client = web_client

    def _esearch(self, db: str, term: str, sort: str, **extra_params: dict[str, Any]) -> Any:
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
    "OA",
    "can_access",
    "license",
    "link",
}
PAPER_FULL_TEXT_PROPS = {
    "fulltext_xml",
}


class NcbiLookupClient(
    NcbiClientBase,
    IPaperLookupClient,
    IGeneLookupClient,
    IVariantLookupClient,
    IAnnotateEntities,
):
    """A client for querying the various services in the NCBI API."""

    # According to https://support.nlm.nih.gov/knowledgebase/article/KA-05316/en-us the max
    # RPS for NCBI API endpoints is 3 without an API key, and 10 with an API key.
    SYMBOL_GET_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/gene/symbol/{symbols}/taxon/Human"
    PMCOA_GET_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
    PUBTATOR_GET_URL = (
        "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/pmc_export/bioc{fmt}?pmcids={id}"
    )
    BIOC_GET_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{pmcid}/ascii"

    def __init__(
        self,
        web_client: IWebContentClient,
        settings: dict[str, str] | None = None,
        mcp_mirror_dir: str | None = None,
    ) -> None:
        super().__init__(web_client, settings)
        self._pmc_mirror = None

        if pmc_mirror_dir:
            mirror_path = Path(pmc_mirror_dir)
            if not mirror_path.exists():
                raise ValueError(f"PMC mirror directory does not exist: {pmc_mirror_dir}")

            logger.info(f"PMC mirror directory configured: {pmc_mirror_dir}")
            self._pmc_mirror = _PmcMirror(mirror_path)

    def _get_xml_props(self, article: Any) -> dict[str, str]:
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
        return props

    def _get_license_props(self, pmcid: str) -> dict[str, str | bool]:
        """Get the access status for a paper from the PMC OA API."""
        props: dict[str, str | bool] = {
            "can_access": False,
            "license": "unknown",
            "OA": False,
        }
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
                logger.warning(f"Unexpected PMC OA error for PMCID {pmcid}: {err_code}")
        else:
            props["can_access"] = True
            props["license"] = license = record.attrib.get("license", "unknown")
            props["OA"] = True
            if "-ND" in license:
                # If it has a "no derivatives" license, then we don't consider it open access.
                logger.debug(f"PMC OA record found for {pmcid} but has a no-derivatives license: {license}")
                props["can_access"] = False

        return props

    def _get_derived_props(self, props: dict[str, Any]) -> dict[str, str]:
        """Get the derived properties of a paper."""
        derived_props: dict[str, Any] = {}
        derived_props["citation"] = f"{props['first_author']} ({props['pub_year']}) {props['journal']}"
        derived_props["link"] = f"https://pubmed.ncbi.nlm.nih.gov/{props['pmid']}/"
        return derived_props

    def _extract_document_from_bioc(self, content: str, pmcid: str) -> str:
        """Extract document element from BioC XML content."""
        try:
            root = ElementTree.fromstring(content)
            # Find the specific document by PMC ID
            if (doc := root.find(f"./document[id='{pmcid.upper().lstrip('PMC')}']")) is None and (
                doc := root.find(f"./document[id='{pmcid.upper()}']")
            ) is None:
                logger.warning(f"Document {pmcid} not found in BioC content")
                return ""
            return ElementTree.tostring(doc, encoding="unicode")
        except ElementTree.ParseError as e:
            logger.error(f"Failed to parse BioC XML for {pmcid}: {e}")
            return ""

    def _get_full_text_from_mirror(self, pmcid: str) -> str:
        """Attempt to fetch full text from local PMC mirror."""
        if not self._pmc_mirror:
            raise ValueError("PMC mirror not configured")

        content = self._pmc_mirror.fetch_xml(pmcid)
        return self._extract_document_from_bioc(content, pmcid)

    def _get_full_text_online(self, pmcid: str) -> str:
        """Fetch full text from online NCBI endpoint."""
        logger.debug(f"Fetching {pmcid} from online NCBI endpoint")
        try:
            root = self._web_client.get(self.BIOC_GET_URL.format(pmcid=pmcid), content_type="xml")
            content = ElementTree.tostring(root, encoding="unicode")
            return self._extract_document_from_bioc(content, pmcid)
        except (HTTPError, RetryError, ElementTree.ParseError) as e:
            logger.warning(f"Unexpected error fetching BioC entry for {pmcid}: {e}")
            return ""

    def _get_full_text(self, props: dict[str, Any]) -> str:
        """Get the full text of a paper from PMC."""
        pmcid = props["pmcid"]
        if not props["can_access"]:
            logger.debug(f"Cannot fetch full text, paper 'pmcid:{pmcid}' is not in PMC-OA or has unusable license.")
            return ""

        # Try mirror first if available, otherwise fetch online
        if self._pmc_mirror:
            return self._get_full_text_from_mirror(pmcid)
        else:
            return self._get_full_text_online(pmcid)

    # IPaperLookupClient
    def search(self, query: str, **extra_params: dict[str, Any]) -> Sequence[str]:
        root = self._esearch(db="pubmed", term=query, sort="relevance", **extra_params)
        pmids = [id.text for id in root.findall("./IdList/Id") if id.text]
        return pmids

    def fetch(self, paper_id: str, include_fulltext: bool = False) -> Paper | None:
        if (root := self._efetch(db="pubmed", id=paper_id, retmode="xml", rettype="abstract")) is None:
            return None

        if (article := root.find(f"PubmedArticle/MedlineCitation/PMID[.='{paper_id}']/../..")) is None:
            return None

        props: dict[str, Any] = {"id": f"pmid:{paper_id}", "pmid": paper_id}
        props.update(self._get_xml_props(article))
        props.update(self._get_license_props(props["pmcid"]))
        props.update(self._get_derived_props(props))
        assert set(props.keys()) == PAPER_BASE_PROPS, f"Missing properties: {PAPER_BASE_PROPS ^ set(props.keys())}"
        if include_fulltext:
            props["fulltext_xml"] = self._get_full_text(props)
        return Paper(**props)

    # IGeneLookupClient
    def gene_id_for_symbol(self, *symbols: str, allow_synonyms: bool = False) -> dict[str, int]:
        """Query the NCBI gene database for the gene_id for a given collection of `symbols`.

        If `allow_synonyms` is True, then this will attempt to return the most relevant gene_id for each symbol. If
        there are multiple matches to a symbol, the direct match (where the query symbol is the official symbol) will
        be returned. If there are no direct matches, then the first synonym match will be returned.
        """
        url = self.SYMBOL_GET_URL.format(symbols=",".join(symbols))
        root = self._web_client.get(url, content_type="json")
        return _extract_gene_symbols(root.get("reports", []), symbols, allow_synonyms)

    # IVariantLookupClient
    def hgvs_from_rsid(self, *rsids: str) -> dict[str, dict[str, str]]:
        # Provided rsids should be numeric strings prefixed with `rs`.
        if not rsids or not all(rsid.startswith("rs") and rsid[2:].isnumeric() for rsid in rsids):
            raise ValueError("Invalid rsids list - must provide 'rs' followed by a string of numeric characters.")

        uids = {rsid[2:] for rsid in rsids}
        try:
            root = self._efetch(db="snp", id=",".join(uids), retmode="xml", rettype="xml")
        except HTTPError as e:
            logger.warning(f"Unexpected error fetching HGVS data for rsids {','.join(uids)}: {e}")
            return {}

        return {"rs" + uid: _extract_hgvs_from_xml(root, uid) for uid in uids}

    # IAnnotateEntities
    def annotate(self, paper: Paper) -> dict[str, Any]:
        """Annotate the paper with entities from PubTator."""
        if not paper.props.get("can_access", False):
            logger.warning(f"Cannot annotate, paper '{paper}' is not licensed for access.")
            return {}

        url = self.PUBTATOR_GET_URL.format(fmt="json", id=paper.props["pmcid"])
        return self._web_client.get(url, content_type="json")


def _extract_hgvs_from_xml(root: Any, uid: str) -> dict[str, str]:
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
    gene = props.get("GENE", "").split(":")[0]

    # Return a dict with the first occurrence of each value that starts with 'NP_' (hgvs_p), 'NM_' (hgvs_c), or 'NC_'
    # (genomic reference sequences / non-coding variants).
    types = {
        "hgvs_p": lambda x: x.startswith("NP_"),
        "hgvs_c": lambda x: x.startswith("NM_"),
        "hgvs_g": lambda x: x.startswith("NC_"),
    }
    ret_dict = {k: next(filter(match, hgvs)) for k, match in types.items() if (any(map(match, hgvs)))}
    ret_dict["gene"] = gene
    return ret_dict


def _extract_gene_symbols(reports: list[dict], symbols: Sequence[str], allow_synonyms: bool) -> dict[str, int]:
    matches = {g["gene"]["symbol"]: int(g["gene"]["gene_id"]) for g in reports if g["gene"]["symbol"] in symbols}

    if allow_synonyms:
        for missing_symbol in [s for s in symbols if s not in matches]:
            if synonym := next((g["gene"] for g in reports if missing_symbol in g["query"]), None):
                matches[missing_symbol] = int(synonym["gene_id"])

    return matches


def get_ncbi_response_translator() -> Callable[[str, int, str], tuple[int, str]]:
    def translate(url: str, original_status: int, text: str) -> tuple[int, str]:
        """Translate the status code of an NCBI search response in case they improperly reported a server error."""
        if (
            text
            and original_status == 200
            and url.startswith(NcbiClientBase.EUTILS_HOST + NcbiClientBase.EUTILS_SEARCH_SITE)
            and (error := ElementTree.fromstring(text).find("ERROR")) is not None
        ):
            # Extract error code by returning the first occurrence of an integer between 400 and 600 in the error text.
            status = next(
                (int(s) for s in (error.text or "").split() if s.isnumeric() and 400 <= int(s) < 600),
                500,
            )
            logger.warning(f"NCBI esearch request failed with status {status}: {error.text}")
            return status, text
        return original_status, text

    return translate
