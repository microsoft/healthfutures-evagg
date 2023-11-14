from typing import Protocol


class IEntrezClient(Protocol):
    def efetch(self, db: str, id: str, retmode: str | None = None, rettype: str | None = None) -> str:
        """Call the Entrez EFetch API. `id` can be a comma-separated list of IDs."""
        ...
