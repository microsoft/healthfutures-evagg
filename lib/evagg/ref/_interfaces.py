from typing import Dict, Protocol, Sequence


class INcbiSnpClient(Protocol):
    def hgvs_from_rsid(self, rsid: Sequence[str]) -> Dict[str, Dict[str, str]]:
        ...


class INcbiGeneClient(Protocol):
    def gene_id_for_symbol(self, symbols: Sequence[str], allow_synonyms: bool = False) -> Dict[str, int]:
        ...
