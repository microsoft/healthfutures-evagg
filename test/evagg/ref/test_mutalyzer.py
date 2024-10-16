from typing import Any

import pytest
import requests

from lib.evagg.ref import MutalyzerClient
from lib.evagg.utils import IWebContentClient


@pytest.fixture
def mock_web_client(mock_client: Any) -> Any:
    return mock_client(IWebContentClient)


def test_back_translate_success(mock_web_client: Any) -> None:
    web_client = mock_web_client(["NM_001276.4:c.(104G>A)"])
    result = MutalyzerClient(web_client).back_translate("NP_001267.2:p.Arg35Gln")
    assert result == ["NM_001276.4:c.(104G>A)"]


def test_back_translate_failure(mock_web_client: Any) -> None:
    web_client = mock_web_client([])
    result = MutalyzerClient(web_client).back_translate("NP_001267.2:FOO")
    assert result == []


def test_back_translate_frameshift(mock_web_client: Any) -> None:
    web_client = mock_web_client([])
    result = MutalyzerClient(web_client).back_translate("NP_001267.2:p.Arg35fs")
    assert result == []


def test_back_translate_caching(mock_web_client: Any) -> None:
    web_client = mock_web_client(["NM_001276.4:c.(104G>A)"], ["Something else"])
    mutalyzer = MutalyzerClient(web_client)
    result = mutalyzer.back_translate("NP_001267.2:p.Arg35Gln")
    assert result == ["NM_001276.4:c.(104G>A)"]

    result = mutalyzer.back_translate("NP_001267.2:p.Arg35Gln")
    assert result == ["NM_001276.4:c.(104G>A)"]


def test_normalize_success(mock_web_client: Any) -> None:
    web_client = mock_web_client("mutalyzer_normalize_protein.json")
    result = MutalyzerClient(web_client).normalize("NP_001267.2:p.Arg35Gln")
    assert result["normalized_description"] == "NP_001267.2:p.(Arg35Gln)"

    web_client = mock_web_client("mutalyzer_normalize_coding.json")
    result = MutalyzerClient(web_client).normalize("NM_001276.4:c.104G>A")
    assert result["protein"]["description"] == "NM_001276.4(NP_001267.2):p.(Arg35Gln)"

    web_client = mock_web_client([])
    result = MutalyzerClient(web_client).normalize("NP_001267.2:p.A35fs")
    assert result["normalized_description"] == "NP_001267.2:p.Ala35fs"


def test_normalize_failure(mock_web_client: Any) -> None:
    web_client = mock_web_client("mutalyzer_normalize_fail.json")
    result = MutalyzerClient(web_client).normalize("NP_001267.2:FOO")
    assert result.get("error_message") == "ESYNTAXUC"


def test_normalize_caching(mock_web_client: Any) -> None:
    web_client = mock_web_client("mutalyzer_normalize_protein.json", "mutalyzer_normalize_fail.json")
    mutalyzer = MutalyzerClient(web_client)
    result = mutalyzer.normalize("NP_001267.2:p.Arg35Gln")
    assert result["normalized_description"] == "NP_001267.2:p.(Arg35Gln)"

    result = mutalyzer.normalize("NP_001267.2:p.Arg35Gln")
    assert result["normalized_description"] == "NP_001267.2:p.(Arg35Gln)"


def test_normalize_service_error() -> None:

    class ThrowingWebClient(IWebContentClient):
        def __init__(self, error: Exception) -> None:
            self._error = error

        def get(self, url: str, content_type: str) -> Any:
            raise (self._error)

    web_client_other = ThrowingWebClient(TypeError("Some other kind of error"))
    mutalyzer = MutalyzerClient(web_client_other)
    with pytest.raises(TypeError):
        mutalyzer.normalize("NP_001267.2:p.Arg35Gln")

    response_500 = requests.Response()
    response_500.status_code = 500
    web_client_500 = ThrowingWebClient(requests.exceptions.HTTPError(response=response_500))
    mutalyzer = MutalyzerClient(web_client_500)
    result = mutalyzer.normalize("NP_001267.2:p.Arg35Gln")
    assert result == {"error_message": "Mutalyzer system error"}

    response_422 = requests.Response()
    response_422.status_code = 422
    web_client_422 = ThrowingWebClient(requests.exceptions.HTTPError(response=response_422))
    mutalyzer = MutalyzerClient(web_client_422)
    with pytest.raises(requests.exceptions.HTTPError):
        mutalyzer.normalize("NP_001267.2:p.Arg35Gln")


def test_validate_success(mock_web_client: Any) -> None:
    web_client = mock_web_client("mutalyzer_normalize_coding.json")
    validation_result, error_message = MutalyzerClient(web_client).validate("NM_001276.4:c.104G>A")
    assert validation_result
    assert error_message is None


def test_validate_failure(mock_web_client: Any) -> None:
    web_client = mock_web_client([])
    validation_result, error_message = MutalyzerClient(web_client).validate("c.104G>A")
    assert not validation_result
    assert error_message == "Invalid HGVS description"

    web_client = mock_web_client([])
    validation_result, error_message = MutalyzerClient(web_client).validate("NP_001267.2:p.A35fs")
    assert not validation_result
    assert error_message == "Frameshift validation not supported"

    web_client = mock_web_client("mutalyzer_normalize_fail.json")
    validation_result, error_message = MutalyzerClient(web_client).validate("NP_001267.2:FOO")
    assert not validation_result
    assert error_message == "ESYNTAXUC"
