from collections.abc import Sequence
from typing import Any, Protocol


class ISearchPapers(Protocol):
    async def search_papers(self, gene_symbol: str, query_params: dict[str, Any]) -> Sequence[str]:
        """Search for paper IDs (PMIDs) related to gene_symbol with optional query parameters."""
        ...  # pragma: no cover
