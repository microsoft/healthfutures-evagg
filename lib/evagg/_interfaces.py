from typing import Dict, Mapping, Protocol, Sequence, Set

from lib.evagg.types import IPaperQuery, Paper


class IEvAggApp(Protocol):
    def execute(self) -> None:
        ...


class IGetPapers(Protocol):
    def search(self, query: IPaperQuery) -> Set[Paper]:
        ...
        #
        # Set of papers that mention the gene
        #


class IExtractFields(Protocol):
    def extract(self, paper: Paper, query: IPaperQuery) -> Sequence[Dict[str, str]]:
        ...


class IWriteOutput(Protocol):
    def write(self, fields: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
        ...
