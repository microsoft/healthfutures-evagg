import logging
import urllib.parse as urlparse
from typing import Any, Dict, Optional, Sequence

import requests
from defusedxml import ElementTree as ElementTree
from pydantic import root_validator

from lib.config import PydanticYamlModel
from lib.evagg.web.entrez import IEntrezClient

from .interfaces import IGeneLookupClient, IVariantLookupClient

REQUIRED: Dict[str, str] = {
    "api_key": "NCBI_EUTILS_API_KEY",
    "email": "NCBI_EUTILS_EMAIL",
}


class NcbiApiSettings(PydanticYamlModel):
    api_key: str
    email: str
    max_tries: str = "10"

    @root_validator(pre=True)
    @classmethod
    def _validate_required(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in REQUIRED.items():
            if k not in values or not values[k]:
                raise ValueError(f"Missing required setting: {v}")
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


class NcbiGeneClient(IGeneLookupClient):
    def __init__(self, entrez_client: IEntrezClient) -> None:
        self._entrez_client = entrez_client

    @classmethod
    def _get_json(cls, url: str, timeout: int = 10) -> Dict[str, Any]:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        if len(r.content) == 0:
            return {}
        return r.json()

    @classmethod
    def gene_id_for_symbol(cls, symbols: Sequence[str], allow_synonyms: bool = False) -> Dict[str, int]:
        """Query the NCBI gene database for the gene_id for a given collection of `symbols`.

        If `allow_synonyms` is True, then this will attempt to return the most relevant gene_id for each symbol, if
        there are multiple matches to a sybol, the direct match (where the query symbol is the official symbol) will
        be returned. If there are no direct matches, then the first synonym match will be returned.
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        # TODO, wrap in Bio.Entrez library as they're better about rate limiting and such.
        # TODO, caching
        url = f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha/gene/symbol/{','.join(symbols)}/taxon/Human"
        raw = cls._get_json(url)

        return _extract_gene_data(raw, symbols, allow_synonyms)


class NcbiSnpClient(IVariantLookupClient):
    def __init__(self, entrez_client: IEntrezClient) -> None:
        self._entrez_client = entrez_client

    def _entrez_fetch_xml(self, db: str, id_str: str) -> Any | None:
        string_response = self._entrez_client.efetch(db=db, id=id_str, retmode="xml", rettype="xml")
        if len(string_response) == 0:
            return None
        # fromstring returns type Any, but it's actually an ElementTree.Element. Using this approach to avoid
        # making bandit cranky with xml.* imports.
        return ElementTree.fromstring(string_response)

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
        response = self._entrez_fetch_xml(db="snp", id_str=rsids_str)

        if response is None or not _validate_snp_response_xml(response):
            return {}

        result = {}
        for rsid in keys:
            result["rs" + rsid] = _find_hgvs_xml(response, rsid)
        return result


class NcbiLookupClient(IGeneLookupClient, IVariantLookupClient):
    HOST = "https://eutils.ncbi.nlm.nih.gov"
    FETCH_TEMPLATE = "/entrez/eutils/efetch.fcgi?db={db}&id={id}&retmode={retmode}&rettype={rettype}&tool=biopython"
    SEARCH_TEMPLATE = "/entrez/eutils/esearch.fcgi?db={db}&term={term}&sort={sort}&retmax={retmax}&tool=biopython"
    KEY_TEMPLATE = "&email={email}&api_key={key}"

    def __init__(self, settings: Optional[Dict[str, str]] = None) -> None:
        self._config: Optional[NcbiApiSettings] = NcbiApiSettings(**settings) if settings is not None else None

    def _get_key_string(self) -> str:
        if self._config is None:
            return ""
        return self.KEY_TEMPLATE.format(email=urlparse.quote(self._config.email), key=self._config.api_key)

    def efetch(self, db: str, id: str, retmode: str | None = None, rettype: str | None = None) -> str:
        url = self.FETCH_TEMPLATE.format(db=db, id=id, retmode=retmode, rettype=rettype)
        response = requests.get(f"{self.HOST}{url}{self._get_key_string()}")
        return response.text

    def esearch(self, db: str, term: str, sort: str, retmax: int, retmode: str | None = None) -> str:
        url = self.SEARCH_TEMPLATE.format(db=db, term=term, sort=sort, retmax=retmax)
        response = requests.get(f"{self.HOST}{url}{self._get_key_string()}")
        return response.text

    def gene_id_for_symbol(self, symbols: Sequence[str], allow_synonyms: bool = False) -> Dict[str, int]:
        """Query the NCBI gene database for the gene_id for a given collection of `symbols`.

        If `allow_synonyms` is True, then this will attempt to return the most relevant gene_id for each symbol, if
        there are multiple matches to a sybol, the direct match (where the query symbol is the official symbol) will
        be returned. If there are no direct matches, then the first synonym match will be returned.
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        # TODO, wrap in Bio.Entrez library as they're better about rate limiting and such.
        # TODO, caching
        url = f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha/gene/symbol/{','.join(symbols)}/taxon/Human"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        raw = {} if len(r.content) == 0 else r.json()

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
        string_response = self.efetch(db="snp", id=rsids_str, retmode="xml", rettype="xml")
        # fromstring returns type Any, but it's actually an ElementTree.Element. Using this approach to avoid
        # making bandit cranky with xml.* imports.
        response = None if len(string_response) == 0 else ElementTree.fromstring(string_response)

        if response is None or not _validate_snp_response_xml(response):
            return {}

        result = {}
        for rsid in keys:
            result["rs" + rsid] = _find_hgvs_xml(response, rsid)
        return result
