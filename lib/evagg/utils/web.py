import hashlib
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from azure.cosmos import ContainerProxy, CosmosClient
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from defusedxml import ElementTree
from pydantic import Extra, validator
from requests.adapters import HTTPAdapter, Retry

from lib.evagg.utils.settings import SettingsModel

from .interfaces import IWebContentClient

logger = logging.getLogger(__name__)

CONTENT_TYPES = ["text", "json", "xml"]


class WebClientSettings(SettingsModel, extra=Extra.forbid):
    max_retries: int = 0  # no retries by default
    retry_backoff: float = 0.5  # indicates progression of 0.5, 1, 2, 4, 8, etc. seconds
    retry_codes: List[int] = [429, 500, 502, 503, 504]  # rate-limit exceeded, server errors
    no_raise_codes: List[int] = []  # don't raise exceptions for these codes
    content_type: str = "text"
    timeout: float = 15.0  # seconds
    status_code_translator: Optional[Callable[[str, int, str], Tuple[int, str]]] = None

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
        self._get_status_code = self._settings.status_code_translator or (lambda _, c, s: (c, s))

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

    def _raise_for_status(self, code: int) -> None:
        """Raise an exception if the status code is not 2xx."""
        if code >= 400 and code < 600 and code not in self._settings.no_raise_codes:
            response = requests.Response()
            response.status_code = code
            raise requests.HTTPError(f"Request failed with status code {code}", response=response)

    def _transform_content(self, text: str, content_type: Optional[str]) -> Any:
        """Get the content from the response based on the provided content type."""
        content_type = content_type or self._settings.content_type
        if content_type == "text":
            return text
        elif content_type == "json":
            return json.loads(text) if text else {}
        elif content_type == "xml":
            return ElementTree.fromstring(text) if text else None
        else:
            raise ValueError(f"Invalid content type: {content_type}")

    def _get_content(self, url: str) -> Tuple[int, str]:
        """GET the text content at the provided URL."""
        response = self._get_session().get(url, timeout=self._settings.timeout)
        return self._get_status_code(url, response.status_code, response.text)

    def _post_content(self, url: str, data: Dict[str, Any]) -> Tuple[int, str]:
        """POST the data to the provided URL."""
        response = self._get_session().post(url, json=data, timeout=self._settings.timeout)
        return self._get_status_code(url, response.status_code, response.text)

    def update_settings(self, **kwargs: Any) -> None:
        """Update the default values for the session."""
        updated_settings = {**self._settings.dict(), **kwargs}
        self._settings = WebClientSettings(**updated_settings)
        self._session = None  # reset the session to apply the new settings

    def get(self, url: str, content_type: Optional[str] = None, url_extra: Optional[str] = None) -> Any:
        """GET the content at the provided URL."""
        code, content = self._get_content(url + (url_extra or ""))
        self._raise_for_status(code)
        return self._transform_content(content, content_type)

    def post(
        self, url: str, data: Dict[str, Any], content_type: Optional[str] = None, url_extra: Optional[str] = None
    ) -> Any:
        """POST the data to the provided URL."""
        code, content = self._post_content(url + (url_extra or ""), data)
        self._raise_for_status(code)
        return self._transform_content(content, content_type)


class CacheClientSettings(SettingsModel, extra=Extra.forbid):
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
            # Attempt to get the item from the cache.
            item = container.read_item(item=cache_key, partition_key=cache_key)
            logger.debug(f"{item['url']} served from {self._cache_settings.database}/{self._cache_settings.container}.")
            code = item.get("status_code", 200)
        except CosmosResourceNotFoundError:
            # If the item is not in the cache, fetch it from the web.
            code, content = super()._get_content(url + (url_extra or ""))
            item = {"id": cache_key, "url": url, "status_code": code, "content": content}
            # Don't cache the response if it's a retryable/transient error.
            if code not in self._settings.retry_codes:
                container.upsert_item(item)

        self._raise_for_status(code)
        return super()._transform_content(item["content"], content_type)

    def _invariant_hash(self, input: str) -> str:
        return hashlib.sha256(input.encode("utf-8")).hexdigest()

    def post(
        self, url: str, data: Dict[str, Any], content_type: Optional[str] = None, url_extra: Optional[str] = None
    ) -> Any:
        """Post the content at the provided URL, using the cache if available."""
        cache_key = url.removeprefix("http://").removeprefix("https://")
        cache_key = cache_key.replace(":", "|").replace("/", "|").replace("?", "|").replace("#", "|")
        cache_key += f"POST={self._invariant_hash(json.dumps(data))}"

        container = self._get_container()

        try:
            # Attempt to get the item from the cache.
            item = container.read_item(item=cache_key, partition_key=cache_key)
            logger.debug(f"{item['url']} served from {self._cache_settings.database}/{self._cache_settings.container}.")
            code = item.get("status_code", 200)
        except CosmosResourceNotFoundError:
            # If the item is not in the cache, fetch it from the web.
            code, content = super()._post_content(url + (url_extra or ""), data)
            item = {"id": cache_key, "url": url, "status_code": code, "content": content}
            # Don't cache the response if it's a retryable/transient error.
            if code not in self._settings.retry_codes:
                container.upsert_item(item)

        self._raise_for_status(code)
        return super()._transform_content(item["content"], content_type)
