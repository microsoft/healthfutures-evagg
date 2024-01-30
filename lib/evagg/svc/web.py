import logging
import xml.etree.ElementTree as Et
from typing import Any, List, Optional

import requests
from requests.adapters import HTTPAdapter, Retry

from lib.config import PydanticYamlModel

from .interfaces import IWebContentClient

logger = logging.getLogger(__name__)

CONTENT_TYPES = ["text", "binary", "json", "xml"]


def _content_type_check(content_type: Optional[str], default: str) -> str:
    content_type = (content_type or default).lower()
    if content_type not in CONTENT_TYPES:
        raise ValueError(f"Web content type must be one of {'/'.join(CONTENT_TYPES)}, got '{content_type}'")
    return content_type


class WebClientSettings(PydanticYamlModel):
    max_retries: int = 0
    retry_backoff: float = 0.5
    retry_codes: List[int] = [502, 503, 504]
    content_type: str = "text"


class RequestsWebContentClient(IWebContentClient):
    """A web content client that uses the requests/urllib3 libraries."""

    def __init__(self, default_content_type: Optional[str] = None) -> None:
        self._session: Optional[requests.Session] = None
        self._defaults = WebClientSettings(content_type=_content_type_check(default_content_type, "text"))

    def _get_session(self) -> requests.Session:
        """Get the session, initializing it if necessary."""
        if self._session is None:
            self._session = requests.Session()
            retries = Retry(
                total=self._defaults.max_retries,
                backoff_factor=self._defaults.retry_backoff,
                status_forcelist=self._defaults.retry_codes,
            )
            self._session.mount("https://", HTTPAdapter(max_retries=retries))
            self._session.mount("http://", HTTPAdapter(max_retries=retries))
        return self._session

    def _get_content(self, response: requests.Response, content_type: Optional[str]) -> Any:
        """Get the content from the response based on the provided content type."""
        content_type = _content_type_check(content_type, self._defaults.content_type)
        if content_type == "text":
            return response.text
        elif content_type == "binary":
            return response.content
        elif content_type == "json":
            return response.json() if len(response.content) > 0 else {}
        elif content_type == "xml":
            return Et.fromstring(response.content) if len(response.content) > 0 else None
        else:
            raise ValueError(f"Invalid content type: {content_type}")

    def set_session_defaults(self, **kwargs: Any) -> None:
        """Set the default values for the session."""
        for key, value in kwargs.items():
            if key not in self._defaults.dict().keys():
                raise ValueError(f"Invalid session default: {key}")
            setattr(self._defaults, key, value)

    def get(self, url: str, content_type: Optional[str] = None) -> Any:
        """GET the content at the provided URL."""
        session = self._get_session()
        response = session.get(url)

        return self._get_content(response, content_type)
