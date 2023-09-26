from typing import Protocol, Sequence

class IGetPapers(Protocol):
    def search(self, gene: str, variant: str) -> Sequence[dict[str, str]]:
        ...


class IExtractFields(Protocol):
    def extract(self, paper: dict[str, str]) -> dict[str, str]:
        ...
    

class IWriteOutput(Protocol):
    def write(self, fields: dict[str, dict[str, str]]) -> None:
        ...