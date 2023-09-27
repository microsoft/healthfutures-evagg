from typing import Protocol, Sequence

from ._base import Paper, Variant

class IGetPapers(Protocol):
    def search(self, query: Variant) -> Sequence[Paper]:
        ...


class IExtractFields(Protocol):
    def extract(self, paper: Paper) -> Sequence[dict[str, str]]:
        ...
    

class IWriteOutput(Protocol):
    def write(self, fields: dict[str, dict[str, str]]) -> None:
        ...