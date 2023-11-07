import json
from unittest import mock

import pytest
from requests import HTTPError

from lib.evagg.lit.pubmed import PubtatorEntityAnnotator
from lib.evagg.types import Paper


def mocked_requests_get(url, params=None, **kwargs):
    class MockResponse:
        def __init__(self, json_data, status_code):
            self.json_data = json_data
            self.content = json.dumps(json_data).encode("utf-8") if json_data else b""
            self.status_code = status_code

        def json(self):
            return self.json_data

        def raise_for_status(self):
            if self.status_code != 200:
                raise HTTPError()  # type: ignore

    if str(url).endswith("PMC1234567"):
        # this is a known paper
        return MockResponse({"foo": "bar"}, 200)
    else:
        # this is an unknown paper
        return MockResponse(None, 200)


@mock.patch("requests.get", side_effect=mocked_requests_get)
def test_annotation(mocked_get):
    paper = Paper(id="123", pmcid="PMC1234567")
    annotator = PubtatorEntityAnnotator()
    annotations = annotator.annotate(paper)
    assert isinstance(annotations, dict)


@mock.patch("requests.get", side_effect=mocked_requests_get)
def test_failed_annotation(mocked_get):
    paper = Paper(id="123", pmcid="PMC7654321")
    annotator = PubtatorEntityAnnotator()
    annotations = annotator.annotate(paper)
    assert annotations == {}

    paper_no_pmcid = Paper(id="123")
    with pytest.raises(ValueError):
        annotator.annotate(paper_no_pmcid)
