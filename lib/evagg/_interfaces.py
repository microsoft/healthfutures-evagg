from typing import Protocol, Sequence

from ._base import Paper, Query


class IGetPapers(Protocol):
    def search(self, query: Query) -> Sequence[Paper]:
        ...


class IExtractFields(Protocol):
    def extract(self, query: Query, paper: Paper) -> Sequence[dict[str, str]]:
        ...


class IWriteOutput(Protocol):
    def write(self, fields: dict[str, Sequence[dict[str, str]]]) -> None:
        ...
