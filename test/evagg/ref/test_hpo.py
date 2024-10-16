from typing import Any

import pytest

from lib.evagg.ref import PyHPOClient, WebHPOClient
from lib.evagg.utils import IWebContentClient


@pytest.fixture
def mock_web_client(mock_client: Any) -> Any:
    return mock_client(IWebContentClient)


def test_compare() -> None:
    term_near1 = "HP:0000123"  # Nephritis
    term_near2 = "HP:0000124"  # Renal tubular dysfunction
    term_far = "HP:0001166"  # Arachnodactyly

    reference = PyHPOClient()

    assert reference.compare(term_near1, term_near2) > reference.compare(term_near1, term_far)
    assert reference.compare(term_near1, term_near2) > reference.compare(term_near1, term_far)


def test_compare_set() -> None:
    term_near1 = "HP:0000123"  # Nephritis
    term_near2 = "HP:0000124"  # Renal tubular dysfunction
    term_far = "HP:0001166"  # Arachnodactyly

    reference = PyHPOClient()

    result = reference.compare_set([term_near1, term_near2], [term_far])
    assert term_near1 in result and result[term_near1][1] == term_far
    assert term_near2 in result and result[term_near2][1] == term_far

    result = reference.compare_set([term_near1], [term_near2, term_far])
    assert term_near1 in result and result[term_near1][1] == term_near2

    result = reference.compare_set([term_near1], [])
    assert term_near1 in result and result[term_near1][1] == ""


def test_fetch() -> None:
    reference = PyHPOClient()

    assert reference.fetch("Nephritis") == {"id": "HP:0000123", "name": "Nephritis"}


def test_exists() -> None:
    reference = PyHPOClient()

    assert reference.exists("HP:0000123")
    assert not reference.exists("HP:9999999")


def test_search(mock_web_client: Any) -> None:
    web_client = mock_web_client("hpo_search_cardiomegaly.json")
    reference = WebHPOClient(web_client)
    result = reference.search("cardiomegaly")
    print(result)
    assert result == [
        {
            "id": "HP:0001640",
            "name": "Cardiomegaly",
            "definition": (
                "Increased size of the heart, clinically defined as an increased transverse diameter of the cardiac "
                "silhouette that is greater than or equal to 50% of the transverse diameter of the chest (increased "
                "cardiothoracic ratio) on a posterior-anterior projection of a chest radiograph or a computed "
                "tomography."
            ),
            "synonyms": ["Enlarged heart", "Increased heart size"],
        }
    ]

    web_client = mock_web_client("hpo_search_asdf.json")
    reference = WebHPOClient(web_client)
    result = reference.search("asdf")
    assert not result
