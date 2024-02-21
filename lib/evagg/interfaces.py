from typing import Dict, Mapping, Protocol, Sequence, Set

from lib.evagg.types import Paper


class IEvAggApp(Protocol):
    def execute(self) -> None:
        """Execute the application."""
        ...  # pragma: no cover


class IGetPapers(Protocol):
    def search(self, query: str) -> Set[Paper]:
        """Search for papers based on the query."""
        ...  # pragma: no cover


class IExtractFields(Protocol):
    def extract(self, paper: Paper, query: str) -> Sequence[Dict[str, str]]:
        """Extract fields from the paper based on the query."""
        ...  # pragma: no cover


class IWriteOutput(Protocol):
    def write(self, fields: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
        """Write the output."""
        ...  # pragma: no cover
