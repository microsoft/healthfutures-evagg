from typing import Any, Dict, Sequence

import requests
from defusedxml import ElementTree as ElementTree

from lib.evagg.web.entrez import IEntrezClient

from ._interfaces import INcbiGeneClient, INcbiSnpClient


class NcbiGeneClient(INcbiGeneClient):
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


class NcbiSnpClient(INcbiSnpClient):
    def __init__(self, entrez_client: IEntrezClient) -> None:
        self._entrez_client = entrez_client

    def _entrez_fetch(self, db: str, id_str: str, retmode: str | None, rettype: str | None) -> str:
        return self._entrez_client.efetch(db=db, id=id_str, retmode=retmode, rettype=rettype)

    def _entrez_fetch_xml(self, db: str, id_str: str) -> Any | None:
        string_response = self._entrez_fetch(db=db, id_str=id_str, retmode="xml", rettype="xml")
        if len(string_response) == 0:
            return None
        # fromstring returns type Any, but it's actually an ElementTree.Element. Using this approach to avoid
        # making bandit cranky with xml.* imports.
        return ElementTree.fromstring(string_response)

    def _validate_snp_response_xml(self, response: Any) -> bool:
        if not response.tag == "{https://www.ncbi.nlm.nih.gov/SNP/docsum}ExchangeSet":
            return False
        if len(response) < 1:
            return False
        for child in response:
            if not child.tag == "{https://www.ncbi.nlm.nih.gov/SNP/docsum}DocumentSummary":
                return False
        return True

    def _find_hgvs_xml(self, response: Any, id: str) -> Dict[str, str]:
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

        if response is None or not self._validate_snp_response_xml(response):
            return {}

        result = {}
        for rsid in keys:
            result["rs" + rsid] = self._find_hgvs_xml(response, rsid)
        return result
