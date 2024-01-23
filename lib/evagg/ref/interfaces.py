from typing import Dict, Protocol, Sequence


class IVariantLookupClient(Protocol):
    def hgvs_from_rsid(self, rsid: Sequence[str]) -> Dict[str, Dict[str, str]]:
        ...


class IGeneLookupClient(Protocol):
    def gene_id_for_symbol(self, symbols: Sequence[str], allow_synonyms: bool = False) -> Dict[str, int]:
        ...


class IEntrezClient(Protocol):
    def efetch(self, db: str, id: str, retmode: str | None = None, rettype: str | None = None) -> str:
        """Call the Entrez EFetch API. `id` can be a comma-separated list of IDs."""
        ...

    def esearch(self, db: str, term: str, sort: str, retmax: int, retmode: str | None = None) -> str:
        """Call the Entrez ESearch API."""
        ...
