import json
import os
import tempfile
from typing import Any, Dict

import pytest

from lib.evagg import RareDiseaseFileLibrary, RemoteFileLibrary, SimpleFileLibrary
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import Paper

# TODO, test TruthsetFileLibrary, but better.


@pytest.fixture
def mock_paper_client(mock_client: type) -> IPaperLookupClient:
    return mock_client(IPaperLookupClient)


def test_remote_init(mock_paper_client: Any) -> None:
    max_papers = 4
    paper_client = mock_paper_client()
    library = RemoteFileLibrary(paper_client, max_papers)
    assert library._paper_client == paper_client
    assert library._max_papers == max_papers


def test_remote_no_paper(mock_paper_client: Any) -> None:
    paper_client = mock_paper_client([])
    query = {"gene_symbol": "gene", "retmax": 1}
    result = RemoteFileLibrary(paper_client).get_papers(query)
    assert paper_client.last_call("search") == ({"query": "gene"},)
    assert paper_client.call_count() == 1
    assert not result


def test_remote_single_paper(mock_paper_client: Any) -> None:
    paper = Paper(
        id="10.1016/j.ajhg.2016.05.014",
        citation="Makrythanasis et al. (2016) AJHG",
        abstract="We report on five individuals who presented with intellectual disability and other...",
        pmid="27392077",
        pmcid="PMC5005447",
        is_pmc_oa=True,
    )
    paper_client = mock_paper_client(["27392077"], paper)
    query = {"gene_symbol": "gene", "retmax": 1}
    result = RemoteFileLibrary(paper_client).get_papers(query)
    # TODO: go back once fetch is fixed:
    assert paper_client.last_call("search") == ({"query": "gene"},)
    assert paper_client.last_call("fetch") == ("27392077",)
    assert paper_client.call_count() == 2
    assert result and len(result) == 1 and result.pop() == paper


def test_rare_disease_init(mock_paper_client: Any) -> None:
    paper_client = mock_paper_client()
    library = RareDiseaseFileLibrary(paper_client)
    assert library._paper_client == paper_client


def test_rare_disease_extra_params(mock_paper_client: Any) -> None:
    paper_client = mock_paper_client([])
    query = {"gene_symbol": "TP53", "retmax": 9, "mindate": "2020-01-01"}
    result = RareDiseaseFileLibrary(paper_client).get_papers(query)
    print("result", result)
    assert paper_client.last_call("search") == ({"query": "TP53"},)
    assert paper_client.call_count() == 1
    # assert result[3] == set() and len(result) == 4


def test_rare_disease_no_paper(mock_paper_client: Any) -> None:
    paper_client = mock_paper_client([])
    query = {"gene_symbol": "gene", "retmax": 1}
    result = RareDiseaseFileLibrary(paper_client).get_papers(query)
    # TODO: go back once fetch is fixed: assert paper_client.last_call("get_papers") == ({"gene_symbol": "gene", "retmax": 1})
    assert paper_client.call_count() == 1
    assert result[3] == set() and len(result) == 4


def test_rare_disease_single_paper(mock_paper_client: Any) -> None:
    paper = Paper(
        id="10.1016/j.ajhg.2016.05.014",
        citation="Makrythanasis et al. (2016) AJHG",
        abstract="We report on five individuals who presented with intellectual disability and other...",
        pmid="27392077",
        pmcid="PMC5005447",
        is_pmc_oa=True,
    )
    paper_client = mock_paper_client(["27392077"], paper)
    query = {"gene_symbol": "gene", "retmax": 1}
    result = RareDiseaseFileLibrary(paper_client).get_papers(query)
    # TODO: go back once fetch is fixed: assert paper_client.last_call("get_papers") == ({"gene_symbol": "gene", "retmax": 1})
    assert paper_client.last_call("fetch") == ("27392077",)
    assert result and len(result) == 4  # and result.pop() == paper


def _paper_to_dict(paper: Paper) -> Dict[str, Any]:
    return {
        "id": paper.id,
        "evidence": paper.evidence,
        "citation": paper.citation,
        "abstract": paper.abstract,
        "props": paper.props,
    }


def test_simple_search() -> None:
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
        query = {"gene_symbol": "test gene", "retmax": 1}
        results = library.get_papers(query)

        # Check that the correct papers were returned
        assert len(results) == 3

        assert paper1 in results
        assert paper2 in results
        assert paper3 in results
