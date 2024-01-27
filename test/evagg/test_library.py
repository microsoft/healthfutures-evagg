import json
import os
import tempfile
from typing import Any, Dict

import pytest

from lib.evagg import RemoteFileLibrary, SimpleFileLibrary, TruthsetFileLibrary
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import Paper, Query


@pytest.fixture
def mock_paper_client(mock_client):
    return mock_client([IPaperLookupClient])


def test_remote_init(mock_paper_client):
    max_papers = 4
    paper_client = mock_paper_client()
    library = RemoteFileLibrary(paper_client, max_papers)
    assert library._paper_client == paper_client
    assert library._max_papers == max_papers


def test_remote_no_paper(mock_paper_client):
    paper_client = mock_paper_client([])
    result = RemoteFileLibrary(paper_client).search(Query("gene:mutation"))
    assert paper_client.last_call("search") == ({"query": "gene"}, {"max_papers": 5})
    assert paper_client.call_count() == 1
    assert not result


def test_remote_multi_query_fail(mock_paper_client):
    paper_client = mock_paper_client()
    with pytest.raises(NotImplementedError):
        RemoteFileLibrary(paper_client).search(Query(["gene1:mutation1", "gene2:mutation2"]))
    assert paper_client.call_count() == 0


def test_remote_single_paper(mock_paper_client):
    paper = Paper(
        id="10.1016/j.ajhg.2016.05.014",
        citation="Makrythanasis et al. (2016) AJHG",
        abstract="We report on five individuals who presented with intellectual disability and other...",
        pmid="27392077",
        pmcid="PMC5005447",
        is_pmc_oa=True,
    )
    paper_client = mock_paper_client(["27392077"], paper)
    result = RemoteFileLibrary(paper_client).search(Query("gene:mutation"))
    assert paper_client.last_call("search") == ({"query": "gene"}, {"max_papers": 5})
    assert paper_client.last_call("fetch") == ("27392077",)
    assert paper_client.call_count() == 2
    assert result and len(result) == 1 and result.pop() == paper


def _paper_to_dict(paper: Paper) -> Dict[str, Any]:
    return {
        "id": paper.id,
        "evidence": paper.evidence,
        "citation": paper.citation,
        "abstract": paper.abstract,
        "props": paper.props,
    }


def test_simple_search():
    # Create a temporary directory and write some test papers to it
    with tempfile.TemporaryDirectory() as tmpdir:
        paper1 = Paper(id="1", citation="Test Paper 1", abstract="This is a test paper.", pmcid="PMC1234")
        paper2 = Paper(id="2", citation="Test Paper 2", abstract="This is another test paper.", pmcid="PMC1235")
        paper3 = Paper(id="3", citation="Test Paper 3", abstract="This is a third test paper.", pmcid="PMC1236")
        with open(os.path.join(tmpdir, "paper1.json"), "w") as f:
            json.dump(_paper_to_dict(paper1), f)
        with open(os.path.join(tmpdir, "paper2.json"), "w") as f:
            json.dump(_paper_to_dict(paper2), f)
        with open(os.path.join(tmpdir, "paper3.json"), "w") as f:
            json.dump(_paper_to_dict(paper3), f)

        # Create a SimpleFileLibrary instance and search for papers
        library = SimpleFileLibrary(collections=[tmpdir])
        # This should return all papers in the library.
        results = library.search(Query("test gene:test variant"))

        # Check that the correct papers were returned
        assert len(results) == 3

        assert paper1 in results
        assert paper2 in results
        assert paper3 in results


def test_truthset_single_paper():
    library = TruthsetFileLibrary("data/truth_set_small.tsv")
    results = library.search(Query("COQ2:mutation"))
    assert len(results) == 7
