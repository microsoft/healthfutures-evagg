import urllib.parse as urlparse
from typing import Any, Dict, List, Optional, Sequence

from pydantic import root_validator

from lib.config import PydanticYamlModel
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.svc import IWebContentClient

from .interfaces import IGeneLookupClient, IVariantLookupClient


class NcbiApiSettings(PydanticYamlModel):
    api_key: Optional[str] = None
    email: str = "biomedcomp@microsoft.com"
    max_tries: str = "10"

    def get_key_string(self) -> str:
        key_string = ""
        if self.email:
            key_string += f"&email={urlparse.quote(self.email)}"
        if self.api_key:
            key_string += f"&api_key={self.api_key}"
        return key_string

    @root_validator(pre=True)
    @classmethod
    def _validate_settings(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("api_key") and not values.get("email"):
            raise ValueError("If NCBI_EUTILS_API_KEY is specified NCBI_EUTILS_EMAIL is required.")
        return values


class NcbiLookupClient(IPaperLookupClient, IGeneLookupClient, IVariantLookupClient):
    API_LOOKUP_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/gene/symbol/{symbols}/taxon/Human"
    EUTILS_HOST = "https://eutils.ncbi.nlm.nih.gov"
    EUTILS_FETCH_URL = "/entrez/eutils/efetch.fcgi?db={db}&id={id}&retmode={retmode}&rettype={rettype}&tool=biopython"
    EUTILS_SEARCH_URL = "/entrez/eutils/esearch.fcgi?db={db}&term={term}&sort={sort}&retmax={retmax}&tool=biopython"
    # It isn't particularly clear from the documentation, but it looks like
    # we're getting 400s from Entrez endpoints when max_tries is set too low.
    # see https://biopython.org/docs/1.75/api/Bio.Entrez.html

    def __init__(self, web_client: IWebContentClient, settings: Optional[Dict[str, str]] = None) -> None:
        self._config = NcbiApiSettings(**settings) if settings else NcbiApiSettings()
        self._web_client = web_client

    def efetch(self, db: str, id: str, retmode: str | None = None, rettype: str | None = None) -> Any:
        key_string = self._config.get_key_string()
        url = self.EUTILS_FETCH_URL.format(db=db, id=id, retmode=retmode, rettype=rettype)
        return self._web_client.get(f"{self.EUTILS_HOST}{url}{key_string}", content_type=retmode)

    def esearch(self, db: str, term: str, sort: str, retmax: int, retmode: str | None = None) -> Any:
        key_string = self._config.get_key_string()
        url = self.EUTILS_SEARCH_URL.format(db=db, term=term, sort=sort, retmax=retmax)
        return self._web_client.get(f"{self.EUTILS_HOST}{url}{key_string}", content_type=retmode)

    def gene_id_for_symbol(self, symbols: Sequence[str], allow_synonyms: bool = False) -> Dict[str, int]:
        """Query the NCBI gene database for the gene_id for a given collection of `symbols`.

        If `allow_synonyms` is True, then this will attempt to return the most relevant gene_id for each symbol, if
        there are multiple matches to a sybol, the direct match (where the query symbol is the official symbol) will
        be returned. If there are no direct matches, then the first synonym match will be returned.
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        url = self.API_LOOKUP_URL.format(symbols=",".join(symbols))
        root = self._web_client.get(url, content_type="json")
        return _extract_gene_symbols(root.get("reports", []), symbols, allow_synonyms)

    def hgvs_from_rsid(self, rsids: Sequence[str]) -> Dict[str, Dict[str, str]]:
        """Provided rsids should be a list of strings, each of which is a valid rsid, prefixed with `rs`."""
        if isinstance(rsids, str):
            rsids = [rsids]

        uids = set()
        for rsid in rsids:
            if not rsid.startswith("rs") or not rsid[2:].isnumeric():
                raise ValueError(f"Invalid rsid: {rsid}: only rs followed by a string of numeric characters allowed.")
            uids.add(rsid[2:])

        root = self.efetch(db="snp", id=",".join(uids), retmode="xml", rettype="xml")
        return {"rs" + uid: _extract_hgvs_from_xml(root, uid) for uid in uids} if root is not None else {}


def _extract_hgvs_from_xml(root: Any, uid: str) -> Dict[str, str]:
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
