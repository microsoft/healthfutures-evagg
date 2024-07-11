from typing import Any, Dict, Optional, Protocol


class IWebContentClient(Protocol):
    def get(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        content_type: Optional[str] = None,
        url_extra: Optional[str] = None,
    ) -> Any:
        """GET the content at the provided URL."""
        ...  # pragma: no cover
