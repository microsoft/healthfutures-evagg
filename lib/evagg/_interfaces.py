from typing import Protocol, Sequence, Set

from ._base import Paper, Variant


class IPaperQuery(Protocol):
    def terms(self) -> Set[Variant]:
        ...


class IGetPapers(Protocol):
    def search(self, query: IPaperQuery) -> Set[Paper]:
        ...


class IExtractFields(Protocol):
    def extract(self, paper: Paper, query: IPaperQuery) -> Sequence[dict[str, str]]:
        ...


class IWriteOutput(Protocol):
    def write(self, fields: dict[str, Sequence[dict[str, str]]]) -> None:
        ...
