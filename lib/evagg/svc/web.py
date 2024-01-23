import logging
from typing import Any, Optional

import requests

from .interfaces import IWebContentClient

logger = logging.getLogger(__name__)

CONTENT_TYPES = ["text", "binary", "json", "xml"]


def _content_type_check(content_type: Optional[str], default: str) -> str:
    content_type = (content_type or default).lower()
    if content_type not in CONTENT_TYPES:
        raise ValueError(f"Web content type must be one of {'/'.join(CONTENT_TYPES)}, got '{content_type}'")
    return content_type


class RequestsWebContentClient(IWebContentClient):
    """A web content client that uses the requests library."""

    def __init__(self, default_content_type: Optional[str] = None) -> None:
        self._default_content_type = _content_type_check(default_content_type, "text")

    def get(self, url: str, content_type: Optional[str] = None) -> Any:
        """GET the content at the provided URL."""
        content_type = _content_type_check(content_type, self._default_content_type)
        response = requests.get(url)
        response.raise_for_status()

        if content_type == "text":
            return response.text
        elif content_type == "binary":
            return response.content
        elif content_type == "json":
            return response.json()
        elif content_type == "xml":
            return response.content
        else:
            raise ValueError(f"Invalid content type: {content_type}")
