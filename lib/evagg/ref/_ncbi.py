from typing import Any, Dict, Sequence

import requests


class NCBIGeneReference:
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
