from typing import Dict, Protocol, Sequence, Set

from ._base import Paper, Variant


class IEvAggApp(Protocol):
    def execute(self) -> None:
        ...


class IPaperQuery(Protocol):
    def terms(self) -> Set[Variant]:
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
