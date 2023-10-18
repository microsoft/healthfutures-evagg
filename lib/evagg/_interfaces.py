from typing import Dict, Protocol, Sequence, Set

from lib.evagg.types import IPaperQuery, Paper


class IEvAggApp(Protocol):
    def execute(self) -> None:
        ...


class IGetPapers(Protocol):
    def search(self, query: IPaperQuery) -> Set[Paper]:
        ...


class IExtractFields(Protocol):
    def extract(self, paper: Paper, query: IPaperQuery) -> Sequence[Dict[str, str]]:
        ...


class IWriteOutput(Protocol):
    def write(self, fields: Dict[str, Sequence[Dict[str, str]]]) -> None:
        ...
