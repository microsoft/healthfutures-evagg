import json
import logging
import xml.etree.ElementTree as Et
from typing import Any, Dict, List, Optional, Union

import requests
from azure.cosmos import ContainerProxy, CosmosClient
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from pydantic import Extra, validator
from requests.adapters import HTTPAdapter, Retry

from lib.config import PydanticYamlModel

from .interfaces import IWebContentClient

logger = logging.getLogger(__name__)

CONTENT_TYPES = ["text", "json", "xml"]


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

    def _get_content(
        self, text: str, content_type: Optional[str]
    ) -> Optional[Union[str, bytes, Dict[str, Any], Et.Element]]:
        """Get the content from the response based on the provided content type."""
        content_type = content_type or self._settings.content_type
        if content_type == "text":
            return text
        elif content_type == "json":
            return json.loads(text) if text else {}
        elif content_type == "xml":
            return Et.fromstring(text) if text else None
        else:
            raise ValueError(f"Invalid content type: {content_type}")

    def _get_text(self, url: str) -> str:
        """GET the text content at the provided URL."""
        return self._get_session().get(url).text

    def update_settings(self, **kwargs: Any) -> None:
        """Update the default values for the session."""
        updated_settings = {**self._settings.dict(), **kwargs}
        self._settings = WebClientSettings(**updated_settings)
        self._session = None  # reset the session to apply the new settings

    def get(self, url: str, content_type: Optional[str] = None, url_extra: Optional[str] = None) -> Any:
        """GET the content at the provided URL."""
        text = self._get_text(url + (url_extra or ""))
        return self._get_content(text, content_type)


class CacheClientSettings(PydanticYamlModel, extra=Extra.forbid):
    endpoint: str
    credential: Any
    database: str = "document_cache"
    container: str = "cache"


class CosmosCachingWebClient(RequestsWebContentClient):
    """A web content client that uses a lookaside CosmosDB cache."""

    def __init__(self, cache_settings: Dict[str, Any], web_settings: Optional[Dict[str, Any]] = None) -> None:
        self._cache_settings = CacheClientSettings(**cache_settings)
        self._cache = CosmosClient(self._cache_settings.endpoint, self._cache_settings.credential)
        self._container: Optional[ContainerProxy] = None
        super().__init__(settings=web_settings)

    def _get_container(self) -> ContainerProxy:
        if self._container:
            return self._container
        database = self._cache.get_database_client(self._cache_settings.database)
        self._container = database.get_container_client(self._cache_settings.container)
        return self._container

    def get(self, url: str, content_type: Optional[str] = None, url_extra: Optional[str] = None) -> Any:
        """GET the content at the provided URL, using the cache if available."""
        cache_key = url.removeprefix("http://").removeprefix("https://")
        cache_key = cache_key.replace(":", "|").replace("/", "|").replace("?", "|").replace("#", "|")
        container = self._get_container()

        try:
            item = container.read_item(item=cache_key, partition_key=cache_key)
            logger.info(f"{cache_key} served from cache.")
        except CosmosResourceNotFoundError:
            content = super()._get_text(url + (url_extra or ""))
            item = {"id": cache_key, "url": url, "content": content}
            container.upsert_item(item)

        return super()._get_content(item["content"], content_type)