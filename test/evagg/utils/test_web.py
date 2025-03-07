from unittest.mock import MagicMock, patch

import pytest
import requests
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from pytest import raises

from lib.evagg.utils import CosmosCachingWebClient, RequestsWebContentClient


def test_settings():
    web_client = RequestsWebContentClient()
    web_client.update_settings(
        max_retries=1, retry_backoff=2, retry_codes=[500, 429], no_raise_codes=[422], content_type="json"
    )
    settings = web_client._settings.dict()
    assert settings["max_retries"] == 1
    assert settings["retry_backoff"] == 2
    assert settings["retry_codes"] == [500, 429]
    assert settings["no_raise_codes"] == [422]
    assert settings["content_type"] == "json"

    web_client.update_settings(max_retries=10)
    settings = web_client._settings.dict()
    assert settings["max_retries"] == 10
    assert settings["retry_backoff"] == 2

    with raises(ValueError):
        web_client.update_settings(invalid=1)


@patch("requests.sessions.Session.request")
def test_get_content_types(mock_request):
    mock_request.side_effect = [
        MagicMock(status_code=200, text="test"),
        MagicMock(status_code=200, text="<test>1</test>"),
        MagicMock(status_code=200, text='{"test": 1}'),
        MagicMock(status_code=200, text='{"test": 1}'),
    ]

    with raises(ValueError):
        RequestsWebContentClient(settings={"content_type": "binary"})

    web_client = RequestsWebContentClient()
    assert web_client.get("https://any.url/testing", content_type="text", url_extra="&extra") == "test"
    assert mock_request.call_args.args[1] == "https://any.url/testing&extra"
    assert web_client.get("https://any.url/testing", content_type="xml").tag == "test"  # type: ignore
    assert web_client.get("https://any.url/testing", content_type="json") == {"test": 1}
    with raises(ValueError):
        web_client.get("https://any.url/testing", content_type="invalid")


@patch("urllib3.connectionpool.HTTPConnectionPool._get_conn")
def test_retry_succeeded(mock_get_conn):
    mock_get_conn.return_value.getresponse.side_effect = [
        MagicMock(status=500, headers={}, iter_content=lambda _: [b""]),
        MagicMock(status=429, headers={}, iter_content=lambda _: [b""]),
        MagicMock(status=200, headers={}, iter_content=lambda _: [b""]),
    ]

    settings = {"max_retries": 2, "retry_backoff": 0, "retry_codes": [500, 429]}
    web_client = RequestsWebContentClient(settings)
    web_client.get("https://any.url/testing")

    assert mock_get_conn.return_value.request.call_args.args[0] == "GET"
    assert mock_get_conn.return_value.request.call_args.args[1] == "/testing"
    assert len(mock_get_conn.return_value.request.mock_calls) == 3


@patch("urllib3.connectionpool.HTTPConnectionPool._get_conn")
def test_retry_failed(mock_get_conn):
    mock_get_conn.return_value.getresponse.side_effect = [
        MagicMock(status=429, headers={}, iter_content=lambda _: [b""]),
        MagicMock(status=500, headers={}, iter_content=lambda _: [b""]),
    ]

    settings = {"max_retries": 1, "retry_backoff": 0, "retry_codes": [500, 429]}
    web_client = RequestsWebContentClient(settings)
    with raises(requests.exceptions.RetryError):
        web_client.get("https://any.url/testing")


@pytest.fixture
def mock_container(json_load):
    class Container:
        def __init__(self, cache):
            self.cache = cache
            self.hits = []
            self.misses = []
            self.writes = []

        def read_item(self, item, partition_key):
            assert item == partition_key
            if item in self.cache:
                self.hits.append(item)
                return self.cache[item]
            self.misses.append(item)
            raise CosmosResourceNotFoundError()

        def upsert_item(self, item):
            assert item["id"] not in self.cache
            self.cache[item["id"]] = item
            self.writes.append(item)
            return item

    return Container(json_load("cosmos_cache.json"))


@patch("lib.evagg.utils.web.CosmosClient")
def test_cosmos_cache_hit(mock_client, mock_container):
    mock_client.return_value.get_database_client.return_value.get_container_client.return_value = mock_container

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=CPA6&sort=relevance&retmax=1&tool=biopython"  # noqa
    web_client = CosmosCachingWebClient(cache_settings={"endpoint": "http://localhost", "credential": "test"})
    assert web_client.get(url, content_type="xml", url_extra="this doesn't matter").tag == "eSearchResult"
    assert web_client.get(url, content_type="xml").tag == "eSearchResult"

    url = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/gene/symbol/CPA6/taxon/Human"
    assert web_client.get(url, content_type="json", url_extra="extra")["reports"][0]["query"][0] == "CPA6"
    assert web_client.get(url, content_type="json")["reports"][0]["query"][0] == "CPA6"

    assert len(mock_container.misses) == 0
    assert len(mock_container.hits) == 4
    assert len(mock_container.writes) == 0


@patch("requests.sessions.Session.request")
@patch("lib.evagg.utils.web.CosmosClient")
def test_cosmos_cache_miss(mock_client, mock_request, mock_container):
    mock_client.return_value.get_database_client.return_value.get_container_client.return_value = mock_container
    mock_request.side_effect = [
        MagicMock(status_code=200, text='<?xml version="1.0" encoding="UTF-8" ?><eSearchResult>GGG6</eSearchResult>'),
        MagicMock(status_code=200, text='{"reports": [{"query": ["GGG6"]}]}'),
        MagicMock(status_code=422, text='{"error": "invalid query, no throw"}'),
        MagicMock(status_code=500, text="throws, doesn't cache"),
        MagicMock(status_code=400, text="throws, caches"),
        MagicMock(status_code=400, text="throws, doesn't cache"),
        MagicMock(status_code=400, text="throws, doesn't cache"),
    ]

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=GGG6&sort=relevance&retmax=1&tool=biopython"  # noqa
    web_client = CosmosCachingWebClient(cache_settings={"endpoint": "http://localhost", "credential": "test"})
    web_client.update_settings(retry_codes=[500], no_raise_codes=[422])
    assert web_client.get(url, content_type="xml", url_extra="this doesn't matter").tag == "eSearchResult"
    assert web_client.get(url, content_type="xml").tag == "eSearchResult"

    url = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/gene/symbol/GGG6/taxon/Human"
    assert web_client.get(url, content_type="json", url_extra="extra")["reports"][0]["query"][0] == "GGG6"
    assert web_client.get(url, content_type="json")["reports"][0]["query"][0] == "GGG6"

    url = "https://testing.invalid/invalid/422"
    assert web_client.get(url, content_type="json", url_extra="extra")["error"] == "invalid query, no throw"
    assert web_client.get(url, content_type="json", url_extra="extra")["error"] == "invalid query, no throw"
    url = "https://testing.invalid/invalid/500"
    with raises(requests.exceptions.HTTPError):
        web_client.get(url, content_type="json")
    url = "https://testing.invalid/invalid/400-cache"
    with raises(requests.exceptions.HTTPError):
        web_client.get(url, content_type="json")
    with raises(requests.exceptions.HTTPError):
        web_client.get(url, content_type="json")
    url = "https://testing.invalid/invalid/400-no-cache"
    web_client._cache_settings.no_cache_codes = [400]
    with raises(requests.exceptions.HTTPError):
        web_client.get(url, content_type="json")
    with raises(requests.exceptions.HTTPError):
        web_client.get(url, content_type="json")

    assert len(mock_container.misses) == 7
    assert len(mock_container.writes) == 4
    assert len(mock_container.hits) == 4
