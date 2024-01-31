from typing import Any, Dict, Optional, Protocol, Sequence

from lib.evagg.types import Paper


class IVariantLookupClient(Protocol):
    def hgvs_from_rsid(self, *rsids: str) -> Dict[str, Dict[str, str]]:
        """Get HGVS variants for the given rsids."""
        ...


class IGeneLookupClient(Protocol):
    def gene_id_for_symbol(self, *symbols: str, allow_synonyms: bool = False) -> Dict[str, int]:
        """Get gene ids for the given gene symbols."""
        ...


class IPaperLookupClient(Protocol):
    def search(self, query: str, max_papers: Optional[int] = None) -> Sequence[str]:
        """Search the paper database for the given query."""
        ...

    def fetch(self, paper_id: str) -> Optional[Paper]:
        """Fetch the paper with the given id."""
        ...


class IAnnotateEntities(Protocol):
    def annotate(self, paper: Paper) -> Dict[str, Any]:
        """Annotate entities in the paper."""
        ...
