from typing import Any, Optional, Protocol


class IWebContentClient(Protocol):
    def get(self, url: str, content_type: Optional[str] = None, url_extra: Optional[str] = None) -> Any:
        """GET the content at the provided URL."""
        ...
