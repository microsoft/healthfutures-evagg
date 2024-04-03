import json
import os
import tempfile
from typing import Any, Dict

import pytest

from lib.evagg import RareDiseaseFileLibrary, RemoteFileLibrary, SimpleFileLibrary
from lib.evagg.llm import IPromptClient
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import Paper

# TODO, test TruthsetFileLibrary, but better.


@pytest.fixture
def mock_paper_client(mock_client: type) -> IPaperLookupClient:
    return mock_client(IPaperLookupClient)


@pytest.fixture
def mock_llm_client(mock_client: type) -> IPromptClient:
    return mock_client(IPromptClient)


def test_remote_init(mock_paper_client: Any) -> None:
    paper_client = mock_paper_client()
    library = RemoteFileLibrary(paper_client)
    assert library._paper_client == paper_client


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


def test_rare_disease_init(mock_paper_client: Any, mock_llm_client: Any) -> None:
    paper_client = mock_paper_client()
    llm_client = mock_llm_client()
    library = RareDiseaseFileLibrary(paper_client, llm_client)
    assert library._paper_client == paper_client
    assert library._llm_client == llm_client


def test_rare_disease_extra_params(mock_paper_client: Any, mock_llm_client: Any) -> None:
    paper_client = mock_paper_client([])
    llm_client = mock_llm_client()
    query = {"gene_symbol": "gene", "retmax": 9}
    result = RareDiseaseFileLibrary(paper_client, llm_client).get_papers(query)
    assert paper_client.last_call("search") == (
        {"query": "gene"},
        {"retmax": 9},
    )
    assert paper_client.call_count() == 1
    print("result", result)
    # TODO: fix this


def test_rare_disease_no_paper(mock_paper_client: Any, mock_llm_client: Any) -> None:
    paper_client = mock_paper_client([])
    llm_client = mock_llm_client()
    query = {"gene_symbol": "gene", "retmax": 1}
    result = RareDiseaseFileLibrary(paper_client, llm_client).get_papers(query)
    assert paper_client.last_call("search") == (
        {"query": "gene"},
        {"retmax": 1},
    )
    assert paper_client.call_count() == 1
    assert not result


def test_rare_disease_single_paper(mock_paper_client: Any, mock_llm_client: Any, json_load) -> None:
    rare_disease_paper = Paper(**json_load("rare_disease_paper.json"))
    paper_client = mock_paper_client([rare_disease_paper.props["pmid"]], rare_disease_paper)
    llm_client = mock_llm_client()
    llm_client._responses = iter(
        [json.dumps({"paper_category": "rare disease"}), json.dumps({"paper_category": "rare disease"})]
    )
    query = {"gene_symbol": "gene"}
    result = RareDiseaseFileLibrary(paper_client, llm_client).get_papers(query)
    assert paper_client.last_call("search") == ({"query": "gene"},)
    assert paper_client.last_call("fetch") == ("33057194",)
    assert paper_client.call_count() == 2
    assert llm_client.last_call("prompt_file")[1] == {"system_prompt": "Extract field"}
    assert llm_client.last_call("prompt_file")[2] == {
        "params": {
            "abstract": "We report on ...",
            "title": "Evidence for 28 genetic disorders discovered by combining healthcare and research data",
        }
    }
    assert llm_client.last_call("prompt_file")[3] == {"prompt_settings": {"prompt_tag": "paper_category"}}

    assert llm_client.call_count() == 1
    assert len(result) == 1
    assert result and result.pop() == rare_disease_paper


def test_rare_disease_get_papers(mock_paper_client: Any, mock_llm_client: Any, json_load) -> None:
    rare_disease_paper = Paper(**json_load("rare_disease_paper.json"))
    non_rare_disease_paper = Paper(**json_load("non_rare_disease_paper.json"))
    ids = [rare_disease_paper.props["pmid"], non_rare_disease_paper.props["pmid"]]
    paper_client = mock_paper_client(ids, rare_disease_paper, non_rare_disease_paper)
    llm_client = mock_llm_client(
        json.dumps({"paper_category": "rare disease"}),
        json.dumps({"paper_category": "non-rare disease"}),
    )
    query = {"gene_symbol": "gene"}
    result = RareDiseaseFileLibrary(paper_client, llm_client).get_papers(query)
    assert paper_client.last_call("search") == ({"query": "gene"},)
    assert paper_client.last_call("fetch") == (non_rare_disease_paper.props["pmid"],)
    assert llm_client.last_call("prompt_file")[2] == {
        "params": {
            "abstract": "We report on ...",
            "title": "Pancreatic cancer-derived exosomal microRNA-19a induces β-cell dysfunction by targeting ADCY1 and EPAC2",
        }
    }
    assert result and len(result) == 1
    assert result[0] == rare_disease_paper


def test_rare_disease_get_all_papers(mock_paper_client: Any, mock_llm_client: Any, json_load) -> None:
    # TODO test tie-breaking
    rare_disease_paper = Paper(**json_load("rare_disease_paper.json"))
    non_rare_disease_paper = Paper(**json_load("non_rare_disease_paper.json"))
    ids = [rare_disease_paper.props["pmid"], non_rare_disease_paper.props["pmid"]]
    paper_client = mock_paper_client(ids, rare_disease_paper, non_rare_disease_paper)
    llm_client = mock_llm_client(
        json.dumps({"paper_category": "rare disease"}),
        json.dumps({"paper_category": "non-rare disease"}),
    )
    query = {"gene_symbol": "gene"}
    result = RareDiseaseFileLibrary(paper_client, llm_client).get_all_papers(query)
    assert paper_client.last_call("search") == ({"query": "gene"},)
    assert paper_client.last_call("fetch") == ("34512170",)
    assert paper_client.call_count() == 3
    assert result and len(result) == 2
    assert result == [rare_disease_paper, non_rare_disease_paper]


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
