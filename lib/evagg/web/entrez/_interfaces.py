from typing import Protocol

# TODO, this should really accept a list of IDs, and return a list of results.


class IEntrezClient(Protocol):
    def efetch(self, db: str, id: str, retmode: str | None = None, rettype: str | None = None) -> str:
        ...
