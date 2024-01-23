from typing import Protocol


class IWebContentClient(Protocol):
    def get(self, url: str) -> str:
        """GET the content at the provided URL."""
        ...
