from typing import Any

import pytest

from lib.evagg.ref import MutalyzerClient
from lib.evagg.svc import IWebContentClient


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


def test_back_translate_caching(mock_web_client: Any) -> None:
    web_client = mock_web_client(["NM_001276.4:c.(104G>A)"], ["Something else"])
    mutalyzer = MutalyzerClient(web_client)
    result = mutalyzer.back_translate("NP_001267.2:p.Arg35Gln")
    assert result == ["NM_001276.4:c.(104G>A)"]

    result = mutalyzer.back_translate("NP_001267.2:p.Arg35Gln")
    assert result == ["NM_001276.4:c.(104G>A)"]


def test_normalize_success(mock_web_client: Any) -> None:
    web_client = mock_web_client("mutalyzer_normalize.json")
    result = MutalyzerClient(web_client).normalize("NP_001267.2:p.Arg35Gln")
    assert result["input_description"] == "NP_001267.2:p.Arg35Gln"
    assert result["corrected_description"] == "NP_001267.2:p.Arg35Gln"


def test_normalize_failure(mock_web_client: Any) -> None:
    web_client = mock_web_client("mutalyzer_normalize_fail.json")
    result = MutalyzerClient(web_client).normalize("NP_001267.2:FOO")
    assert result["custom"]["input_description"] == "NP_001267.2:FOO"
    assert result["message"] == "Errors encountered. Check the 'custom' field."


def test_normalize_caching(mock_web_client: Any) -> None:
    web_client = mock_web_client("mutalyzer_normalize.json", "mutalyzer_normalize_fail.json")
    mutalyzer = MutalyzerClient(web_client)
    result = mutalyzer.normalize("NP_001267.2:p.Arg35Gln")
    assert result["input_description"] == "NP_001267.2:p.Arg35Gln"
    assert result["corrected_description"] == "NP_001267.2:p.Arg35Gln"

    result = mutalyzer.normalize("NP_001267.2:p.Arg35Gln")
    assert result["input_description"] == "NP_001267.2:p.Arg35Gln"
    assert result["corrected_description"] == "NP_001267.2:p.Arg35Gln"
