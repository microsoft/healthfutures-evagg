from typing import Protocol, Sequence

from ._base import Paper, Variant


class IPaperQuery(Protocol):
    def terms(self) -> Sequence[Variant]:
        ...


class IGetPapers(Protocol):
    def search(self, query: IPaperQuery) -> Sequence[Paper]:
        ...


class IExtractFields(Protocol):
    def extract(self, paper: Paper, query: IPaperQuery) -> Sequence[dict[str, str]]:
        ...


class IWriteOutput(Protocol):
    def write(self, fields: dict[str, Sequence[dict[str, str]]]) -> None:
        ...
