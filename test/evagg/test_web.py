import json
from unittest.mock import MagicMock, patch

import requests
from pytest import raises

from lib.evagg.svc import RequestsWebContentClient


def test_settings():
    web_client = RequestsWebContentClient()
    web_client.update_settings(max_retries=1, retry_backoff=2, retry_codes=[500, 429], content_type="json")
    settings = web_client._settings.dict()
    assert settings["max_retries"] == 1
    assert settings["retry_backoff"] == 2
    assert settings["retry_codes"] == [500, 429]
    assert settings["content_type"] == "json"

    web_client.update_settings(max_retries=10)
    settings = web_client._settings.dict()
    assert settings["max_retries"] == 10
    assert settings["retry_backoff"] == 2

    with raises(ValueError):
        web_client.update_settings(invalid=1)


def test_content_types():
    web_client = RequestsWebContentClient()
    assert web_client._get_content(MagicMock(text="test"), "text") == "test"
    assert web_client._get_content(MagicMock(content=b"<test>1</test>"), "xml").tag == "test"
    assert web_client._get_content(
        MagicMock(content=b'{"test": 1}', json=lambda: json.loads('{"test": 1}')), "json"
    ) == {"test": 1}

    with raises(ValueError):
        RequestsWebContentClient(settings={"content_type": "invalid"})


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
