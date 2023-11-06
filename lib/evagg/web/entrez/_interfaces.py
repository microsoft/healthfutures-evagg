from typing import Protocol


class IEntrezClient(Protocol):
    def efetch(self, db: str, id: str, retmode: str | None = None, rettype: str | None = None) -> str:
        ...
