from typing import Dict, Protocol, Sequence


class INcbiSnpClient(Protocol):
    def hgvs_from_rsid(self, rsid: str) -> Dict[str, str | None]:
        ...


class INcbiGeneClient(Protocol):
    def gene_id_for_symbol(self, symbols: Sequence[str], allow_synonyms: bool = False) -> Dict[str, int]:
        ...
