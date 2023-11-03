import json
from typing import Any, Dict, Sequence

import requests

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

    def _entrez_fetch_json(self, db: str, id_str: str) -> Dict[str, Any]:
        string_response = self._entrez_fetch(db=db, id_str=id_str, retmode="json", rettype="json")
        if len(string_response) == 0:
            return {}
        return json.loads(string_response)

    def _validate_snp_response(self, response: Dict[str, Any]) -> bool:
        if "primary_snapshot_data" not in response:
            return False
        if "placements_with_allele" not in response["primary_snapshot_data"]:
            return False
        if len(response["primary_snapshot_data"]["placements_with_allele"]) == 0:
            return False
        return True

    def _find_alt_allele(self, placement: Dict[str, Any]) -> Dict[str, Any]:
        if len(placement["alleles"]) > 2:
            print(f"WARNING: Multiple alleles listed for {placement['seq_id']}. Using first alternate.")
        return placement["alleles"][1]

    def _find_hgvsc(self, response: Dict[str, Any]) -> str | None:
        mrna_seqs = [
            p
            for p in response["primary_snapshot_data"]["placements_with_allele"]
            if p["placement_annot"]["mol_type"] == "rna"
        ]

        if len(mrna_seqs) == 0:
            print(f"WARNING: No RNA sequences found for variant {response['refsnp_id']}.")
            return None

        # Prioritize any MANE select transcripts.
        if "mane_select_ids" in response and len(response["mane_select_ids"]) > 0:
            if len(response["mane_select_ids"]) > 1:
                print(f"WARNING: Multiple MANE select transcripts for {response['refsnp_id']}.")
            mane_select = response["mane_select_ids"][0]
            mane_select_seqs = [p for p in mrna_seqs if p["seq_id"] == mane_select]
            if len(mane_select_seqs) == 0:
                print(
                    f"WARNING: MANE select transcripts listed for variant {response['refsnp_id']}",
                    "but none found in sequences.",
                )
            else:
                if len(mane_select_seqs) > 1:
                    print(f"WARNING: Same MANE transcript sequence listed multiple times for {response['refsnp_id']}.")
                return self._find_alt_allele(mane_select_seqs[0])["hgvs"]

        # Otherwise, just use the first RNA sequence.
        return self._find_alt_allele(mrna_seqs[0])["hgvs"]

    def _find_hgvsp(self, response: Dict[str, Any]) -> str | None:
        protein_seqs = [
            p
            for p in response["primary_snapshot_data"]["placements_with_allele"]
            if p["placement_annot"]["mol_type"] == "protein"
        ]

        if len(protein_seqs) == 0:
            print(f"WARNING: No protein sequences found for variant {response['refsnp_id']}.")
            return None

        # Otherwise, just use the first protein sequence.
        return self._find_alt_allele(protein_seqs[0])["hgvs"]

    def hgvs_from_rsid(self, rsid: str) -> Dict[str, str | None]:
        if rsid.startswith("rs"):
            rsid = rsid[2:]

        # Remaining rsid must be only numeric.
        if not rsid.isnumeric():
            return {}

        print(f"hgvs query for {rsid}")
        response = self._entrez_fetch_json(db="snp", id_str=rsid)

        if not self._validate_snp_response(response):
            return {}

        return {"hgvsc": self._find_hgvsc(response), "hgvsp": self._find_hgvsp(response)}
