from typing import Any, Dict, Mapping, Protocol, Sequence

from lib.evagg.types import Paper


class IEvAggApp(Protocol):
    def execute(self) -> None:
        """Execute the application."""
        ...  # pragma: no cover


class IGetPapers(Protocol):
    def get_papers(self, query: Dict[str, Any]) -> Sequence[Paper]:
        """Search for papers based on the query."""
        ...  # pragma: no cover


class IExtractFields(Protocol):
    def extract(self, paper: Paper, gene_symbol: str) -> Sequence[Dict[str, str]]:
        """Extract fields from the paper based on the gene_symbol."""
        ...  # pragma: no cover


class IWriteOutput(Protocol):
    def write(self, fields: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
        """Write the output."""
        ...  # pragma: no cover
