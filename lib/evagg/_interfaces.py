from typing import Protocol, Sequence, Set

from lib.evagg.types import IPaperQuery, Paper


class IGetPapers(Protocol):
    def search(self, query: IPaperQuery) -> Set[Paper]:
        ...


class IExtractFields(Protocol):
    def extract(self, paper: Paper, query: IPaperQuery) -> Sequence[dict[str, str]]:
        ...


class IWriteOutput(Protocol):
    def write(self, fields: dict[str, Sequence[dict[str, str]]]) -> None:
        ...
