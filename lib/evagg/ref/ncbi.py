import urllib.parse as urlparse
from typing import Any, Dict, Optional, Sequence

from pydantic import root_validator

from lib.config import PydanticYamlModel
from lib.evagg.ref import IEntrezClient
from lib.evagg.svc import IWebContentClient

from .interfaces import IGeneLookupClient, IVariantLookupClient


class NcbiApiSettings(PydanticYamlModel):
    api_key: Optional[str] = None
    email: str = "biomedcomp@microsoft.com"
    max_tries: str = "10"

    @root_validator(pre=True)
    @classmethod
    def _validate_settings(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("api_key") and not values.get("email"):
            raise ValueError("If NCBI_EUTILS_API_KEY is specified NCBI_EUTILS_EMAIL is required.")
        return values


def _validate_snp_response_xml(response: Any) -> bool:
    if not response.tag == "{https://www.ncbi.nlm.nih.gov/SNP/docsum}ExchangeSet":
        return False
    if len(response) < 1:
        return False
    for child in response:
        if not child.tag == "{https://www.ncbi.nlm.nih.gov/SNP/docsum}DocumentSummary":
            return False
    return True


def _find_hgvs_xml(response: Any, id: str) -> Dict[str, str]:
    result = {}

    for child in response:
        if not child.get("uid") == id:
            continue

        for nested_child in child:
            if not nested_child.tag == "{https://www.ncbi.nlm.nih.gov/SNP/docsum}DOCSUM":
                continue

            if not nested_child.text:
                break

            values = {}
            for tok in nested_child.text.split("|"):
                eq_delim = tok.split("=")
                if len(eq_delim) != 2:
                    continue
                values[eq_delim[0]] = eq_delim[1]

            if "HGVS" not in values:
                break

            hgvs_values = values["HGVS"].split(",")

            # Just take the first one.
            hgvs_p = next((v for v in hgvs_values if v.startswith("NP_")), None)
            hgvs_c = next((v for v in hgvs_values if v.startswith("NM_")), None)

            if hgvs_p:
                result["hgvs_p"] = hgvs_p
            if hgvs_c:
                result["hgvs_c"] = hgvs_c
            break
    return result


def _extract_gene_data(raw: Dict[str, Any], symbols: Sequence[str], allow_synonyms: bool) -> Dict[str, int]:
    if "reports" not in raw:
        return {}

    if allow_synonyms:
        result: Dict[str, int] = {}

        for g in raw["reports"]:
            if g["gene"]["symbol"] in symbols:
                result[g["gene"]["symbol"]] = int(g["gene"]["gene_id"])

        missing_symbols = [s for s in symbols if s not in result.keys()]
        for symbol in missing_symbols:
            # find the first query match.
            for g in raw["reports"]:
                if symbol in g["query"]:
                    result[symbol] = int(g["gene"]["gene_id"])
                    break
    else:
        result = {
            g["gene"]["symbol"]: int(g["gene"]["gene_id"]) for g in raw["reports"] if g["gene"]["symbol"] in symbols
        }

    return result


class NcbiLookupClient(IEntrezClient, IGeneLookupClient, IVariantLookupClient):
    API_HOST = "https://eutils.ncbi.nlm.nih.gov"
    API_LOOKUP_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/gene/symbol/{symbols}/taxon/Human"
    UTILS_HOST = "https://eutils.ncbi.nlm.nih.gov"
    UTILS_FETCH_URL = "/entrez/eutils/efetch.fcgi?db={db}&id={id}&retmode={retmode}&rettype={rettype}&tool=biopython"
    UTILS_SEARCH_URL = "/entrez/eutils/esearch.fcgi?db={db}&term={term}&sort={sort}&retmax={retmax}&tool=biopython"
    # It isn't particularly clear from the documentation, but it looks like
    # we're getting 400s from Entrez endpoints when max_tries is set too low.
    # see https://biopython.org/docs/1.75/api/Bio.Entrez.html

    def __init__(self, web_client: IWebContentClient, settings: Optional[Dict[str, str]] = None) -> None:
        self._config: Optional[NcbiApiSettings] = NcbiApiSettings(**settings) if settings is not None else None
        self._web_client = web_client

    def _get_key_string(self) -> str:
        if self._config is None:
            return ""
        key_string = ""
        if self._config.email:
            key_string += f"&email={urlparse.quote(self._config.email)}"
        if self._config.api_key:
            key_string += f"&api_key={self._config.api_key}"
        return key_string

    def efetch(self, db: str, id: str, retmode: str | None = None, rettype: str | None = None) -> Any:
        url = self.UTILS_FETCH_URL.format(db=db, id=id, retmode=retmode, rettype=rettype)
        return self._web_client.get(f"{self.UTILS_HOST}{url}{self._get_key_string()}", content_type=retmode)

    def esearch(self, db: str, term: str, sort: str, retmax: int, retmode: str | None = None) -> Any:
        url = self.UTILS_SEARCH_URL.format(db=db, term=term, sort=sort, retmax=retmax)
        return self._web_client.get(f"{self.UTILS_HOST}{url}{self._get_key_string()}", content_type=retmode)

    def symbol_lookup(self, symbols: Sequence[str]) -> Dict[str, Any]:
        url = self.API_LOOKUP_URL.format(symbols=",".join(symbols))
        return self._web_client.get(url, content_type="json")

    def gene_id_for_symbol(self, symbols: Sequence[str], allow_synonyms: bool = False) -> Dict[str, int]:
        """Query the NCBI gene database for the gene_id for a given collection of `symbols`.

        If `allow_synonyms` is True, then this will attempt to return the most relevant gene_id for each symbol, if
        there are multiple matches to a sybol, the direct match (where the query symbol is the official symbol) will
        be returned. If there are no direct matches, then the first synonym match will be returned.
        """
        symbols = [symbols] if isinstance(symbols, str) else symbols
        raw = self.symbol_lookup(symbols)
        return _extract_gene_data(raw, symbols, allow_synonyms)

    def hgvs_from_rsid(self, rsids: Sequence[str]) -> Dict[str, Dict[str, str]]:
        """Provided rsids should be a list of strings, each of which is a valid rsid, prefixed with `rs`."""
        if isinstance(rsids, str):
            rsids = [rsids]

        keys = set()
        for rsid in rsids:
            if not rsid.startswith("rs"):
                raise ValueError(f"Invalid rsid: {rsid}. Did you forget to include an 'rs' prefix?")
            rsid = rsid[2:]

            # Remaining rsid must be only numeric.
            if not rsid.isnumeric():
                raise ValueError(f"Invalid rsid: {rsid}: only rs followed by a string of numeric characters allowed.")

            keys.add(rsid)

        rsids_str = ",".join(keys)
        response = self.efetch(db="snp", id=rsids_str, retmode="xml", rettype="xml")

        if response is None or not _validate_snp_response_xml(response):
            return {}

        result = {}
        for rsid in keys:
            result["rs" + rsid] = _find_hgvs_xml(response, rsid)
        return result
