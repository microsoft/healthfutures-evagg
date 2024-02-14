from typing import Dict, Mapping, Protocol, Sequence, Set

from lib.evagg.types import Paper


class IEvAggApp(Protocol):
    def execute(self) -> None:
        """Execute the application."""
        ...


class IGetPapers(Protocol):
    def search(self, query: str) -> Set[Paper]:
        ...
        #
        # Set of papers that mention the gene
        #


class IExtractFields(Protocol):
    def extract(self, paper: Paper, query: str) -> Sequence[Dict[str, str]]:
        """Extract fields from the paper based on the query."""
        ...


class IWriteOutput(Protocol):
    def write(self, fields: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
        """Write the output."""
        ...
