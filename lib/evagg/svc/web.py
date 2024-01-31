import logging
import xml.etree.ElementTree as Et
from typing import Any, Dict, List, Optional

import requests
from pydantic import Extra, validator
from requests.adapters import HTTPAdapter, Retry

from lib.config import PydanticYamlModel

from .interfaces import IWebContentClient

logger = logging.getLogger(__name__)

CONTENT_TYPES = ["text", "binary", "json", "xml"]


class WebClientSettings(PydanticYamlModel, extra=Extra.forbid):
    max_retries: int = 0  # no retries by default
    retry_backoff: float = 0.5  # indicates progression of 0.5, 1, 2, 4, 8, etc. seconds
    retry_codes: List[int] = [429, 500, 502, 503, 504]  # rate-limit exceeded, server errors
    content_type: str = "text"

    @validator("content_type")
    @classmethod
    def _validate_content_type(cls, value: str) -> str:
        if value not in CONTENT_TYPES:
            raise ValueError(f"Web content type must be one of {'/'.join(CONTENT_TYPES)}, got '{value}'")
        return value


class RequestsWebContentClient(IWebContentClient):
    """A web content client that uses the requests/urllib3 libraries."""

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        self._settings = WebClientSettings(**settings) if settings else WebClientSettings()
        self._session: Optional[requests.Session] = None

    def _get_session(self) -> requests.Session:
        """Get the session, initializing it if necessary."""
        if self._session is None:
            self._session = requests.Session()
            retries = Retry(
                total=self._settings.max_retries,
                backoff_factor=self._settings.retry_backoff,
                status_forcelist=self._settings.retry_codes,
            )
            self._session.mount("https://", HTTPAdapter(max_retries=retries))
            self._session.mount("http://", HTTPAdapter(max_retries=retries))
        return self._session

    def _get_content(self, response: requests.Response, content_type: Optional[str]) -> Any:
        """Get the content from the response based on the provided content type."""
        content_type = content_type or self._settings.content_type
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

    def update_settings(self, **kwargs: Any) -> None:
        """Update the default values for the session."""
        updated_settings = {**self._settings.dict(), **kwargs}
        self._settings = WebClientSettings(**updated_settings)

    def get(self, url: str, content_type: Optional[str] = None) -> Any:
        """GET the content at the provided URL."""
        session = self._get_session()
        response = session.get(url)

        return self._get_content(response, content_type)
