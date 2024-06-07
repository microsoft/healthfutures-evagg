import json
import os
import tempfile
from typing import Any, Dict

import pytest

from lib.evagg import RareDiseaseFileLibrary, SimpleFileLibrary
from lib.evagg.llm import IPromptClient
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import Paper

# TODO, test TruthsetFileHandler, but better.


@pytest.fixture
def mock_paper_client(mock_client: type) -> IPaperLookupClient:
    return mock_client(IPaperLookupClient)


@pytest.fixture
def mock_llm_client(mock_client: type) -> IPromptClient:
    return mock_client(IPromptClient)


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
        [
            json.dumps({"disease_category": "rare disease"}),
            json.dumps({"disease_category": "rare disease"}),
            json.dumps({"disease_category": "rare disease"}),
        ]
    )
    query = {"gene_symbol": "gene"}
    allowed_categories = ["rare disease", "other"]
    result = RareDiseaseFileLibrary(paper_client, llm_client, allowed_categories).get_papers(query)
    print("result", result)
    assert paper_client.last_call("search") == ({"query": "gene"},)
    assert paper_client.last_call("fetch") == ("37187958", {"include_fulltext": True})
    assert paper_client.call_count() == 2
    assert llm_client.last_call("prompt_file")[1] == {"system_prompt": "Extract field"}
    assert llm_client.last_call("prompt_file")[2]["params"]["abstract"] == "The endoplasmic reticulum ..."
    assert llm_client.last_call("prompt_file")[3] == {
        "prompt_settings": {"prompt_tag": "paper_category", "temperature": 0.8},
    }
    assert llm_client.call_count() == 3
    assert len(result) == 1
    assert result and result[0] == rare_disease_paper


def test_rare_disease_get_papers(mock_paper_client: Any, mock_llm_client: Any, json_load) -> None:
    rare_disease_paper = Paper(**json_load("rare_disease_paper.json"))
    other_paper = Paper(**json_load("other_paper.json"))
    ids = [rare_disease_paper.props["pmid"], other_paper.props["pmid"]]
    paper_client = mock_paper_client(ids, rare_disease_paper, other_paper)
    allowed_categories = ["rare disease", "other"]
    llm_client = mock_llm_client(
        json.dumps({"paper_category": "rare disease"}),
        json.dumps({"paper_category": "rare disease"}),
        json.dumps({"paper_category": "rare disease"}),
        json.dumps({"paper_category": "other"}),
        json.dumps({"paper_category": "other"}),
        json.dumps({"paper_category": "other"}),
    )
    query = {"gene_symbol": "gene"}
    result = RareDiseaseFileLibrary(paper_client, llm_client, allowed_categories).get_papers(query)
    assert paper_client.last_call("search") == ({"query": "gene"},)
    assert paper_client.last_call("fetch") == (other_paper.props["pmid"], {"include_fulltext": True})
    assert llm_client.last_call("prompt_file")[2]["params"]["abstract"] == "The endoplasmic reticulum ..."
    assert result and len(result) == 2
    assert result[0] == rare_disease_paper

    # Remove all paper_finding_few_shot_*.txt from lib/evagg/content/prompts
    for file in os.listdir("lib/evagg/content/prompts"):
        if file.startswith("paper_finding_few_shot_"):
            os.remove(os.path.join("lib/evagg/content/prompts", file))


async def test_rare_disease_get_all_papers(mock_paper_client: Any, mock_llm_client: Any, json_load) -> None:
    rare_disease_paper = Paper(**json_load("rare_disease_paper.json"))
    other_paper = Paper(**json_load("other_paper.json"))
    ids = [rare_disease_paper.props["pmid"], other_paper.props["pmid"]]
    paper_client = mock_paper_client(ids, rare_disease_paper, other_paper)
    llm_client = mock_llm_client(
        json.dumps({"paper_category": "rare disease"}),
        json.dumps({"paper_category": "rare disease"}),
        json.dumps({"paper_category": "rare disease"}),
        json.dumps({"paper_category": "other"}),
        json.dumps({"paper_category": "other"}),
        json.dumps({"paper_category": "other"}),
    )
    query = {"gene_symbol": "gene"}
    result = await RareDiseaseFileLibrary(paper_client, llm_client)._get_all_papers(query)
    assert paper_client.last_call("search") == ({"query": "gene"},)
    assert paper_client.last_call("fetch") == ("34512170", {"include_fulltext": True})
    assert paper_client.call_count() == 3
    assert result and len(result) == 2
    assert result == [rare_disease_paper, other_paper]

    # Remove all paper_finding_few_shot_*.txt from lib/evagg/content/prompts
    for file in os.listdir("lib/evagg/content/prompts"):
        if file.startswith("paper_finding_few_shot_"):
            os.remove(os.path.join("lib/evagg/content/prompts", file))


def _paper_to_dict(paper: Paper) -> Dict[str, Any]:
    return {
        "id": paper.id,
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
