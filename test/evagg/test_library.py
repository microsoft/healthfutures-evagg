import json
import os
import tempfile
from typing import Any, Dict
from unittest.mock import patch

import pytest

from lib.evagg import RareDiseaseFileLibrary, SimpleFileLibrary, PaperListLibrary
from lib.evagg.library import RareDiseaseLibraryCached
from lib.evagg.llm import IPromptClient
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import Paper


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

    with pytest.raises(ValueError):
        RareDiseaseFileLibrary(paper_client, llm_client, ["invalid category"])


def test_rare_disease_single_paper(mock_paper_client: Any, mock_llm_client: Any, json_load: Any) -> None:
    rare_disease_paper = Paper(**json_load("rare_disease_paper.json"))
    paper_client = mock_paper_client([rare_disease_paper.props["pmid"]], rare_disease_paper)
    llm_client = mock_llm_client()
    llm_client._responses = iter("genetic disease")
    query = {"gene_symbol": "gene"}
    allowed_categories = ["genetic disease", "other"]
    result = RareDiseaseFileLibrary(paper_client, llm_client, allowed_categories).get_papers(query)
    print("result", result)
    assert len(result) == 1
    assert result[0] == rare_disease_paper


def test_rare_disease_extra_params(mock_paper_client: Any, mock_llm_client: Any, json_load: Any) -> None:
    rare_disease_paper = Paper(**json_load("rare_disease_paper.json"))
    paper_client = mock_paper_client([rare_disease_paper.props["pmid"]], rare_disease_paper)

    # Simple retmax test.
    query = {"gene_symbol": "gene", "retmax": 9}
    result = RareDiseaseFileLibrary(paper_client, mock_llm_client("genetic disease")).get_papers(query)
    assert paper_client.last_call("search") == (
        {"query": "gene pubmed pmc open access[filter]"},
        {"retmax": 9},
    )
    assert paper_client.call_count() == 2
    assert len(result) == 1 and result[0] == rare_disease_paper

    # Simple min date test.
    paper_client = mock_paper_client([rare_disease_paper.props["pmid"]], rare_disease_paper)
    query = {"gene_symbol": "gene", "min_date": "2021/01/01"}
    result = RareDiseaseFileLibrary(paper_client, mock_llm_client("genetic disease")).get_papers(query)
    assert len(result) == 1 and result[0] == rare_disease_paper

    # Retmax results, signifies search overrun, results still returned.
    paper_client = mock_paper_client(
        [rare_disease_paper.props["pmid"], rare_disease_paper.props["pmid"]], rare_disease_paper, rare_disease_paper
    )
    query = {"gene_symbol": "gene", "retmax": 2}
    result = RareDiseaseFileLibrary(paper_client, mock_llm_client("genetic disease", "genetic disease")).get_papers(
        query
    )
    assert len(result) == 2 and result[0] == rare_disease_paper and result[1] == rare_disease_paper


def test_rare_disease_no_paper(mock_paper_client: Any, mock_llm_client: Any) -> None:
    paper_client = mock_paper_client([])
    llm_client = mock_llm_client()
    query = {"gene_symbol": "gene", "retmax": 1}
    result = RareDiseaseFileLibrary(paper_client, llm_client).get_papers(query)
    assert paper_client.last_call("search") == (
        {"query": "gene pubmed pmc open access[filter]"},
        {"retmax": 1},
    )
    assert paper_client.call_count() == 1
    assert not result


def test_rare_disease_paper_without_text(mock_paper_client: Any, mock_llm_client: Any, json_load: Any) -> None:
    rare_disease_paper = Paper(**json_load("rare_disease_paper.json"))
    rare_disease_paper.props["title"] = None
    rare_disease_paper.props["abstract"] = None
    paper_client = mock_paper_client([rare_disease_paper.props["pmid"]], rare_disease_paper)
    llm_client = mock_llm_client("other")
    query = {"gene_symbol": "gene"}
    allowed_categories = ["genetic disease", "other"]
    result = RareDiseaseFileLibrary(paper_client, llm_client, allowed_categories).get_papers(query)
    assert len(result) == 1
    assert result[0] == rare_disease_paper


def test_rare_disease_paper_suffixed_keyword(mock_paper_client: Any, mock_llm_client: Any, json_load: Any) -> None:
    rare_disease_paper = Paper(**json_load("rare_disease_paper.json"))
    rare_disease_paper.props["title"] = "A paper about Bradycardia"
    rare_disease_paper.props["abstract"] = None
    paper_client = mock_paper_client([rare_disease_paper.props["pmid"]], rare_disease_paper)
    llm_client = mock_llm_client("genetic disease")
    query = {"gene_symbol": "gene"}
    allowed_categories = ["genetic disease", "other"]
    result = RareDiseaseFileLibrary(paper_client, llm_client, allowed_categories).get_papers(query)
    assert len(result) == 1
    assert result[0] == rare_disease_paper


def test_rare_disease_paper_incomplete_query(mock_paper_client: Any, mock_llm_client: Any) -> None:
    paper_client = mock_paper_client()
    llm_client = mock_llm_client()
    query: Dict[str, str] = {}
    with pytest.raises(ValueError):
        RareDiseaseFileLibrary(paper_client, llm_client).get_papers(query)

    query = {"gene_symbol": "gene", "max_date": "2021/01/01"}
    with pytest.raises(ValueError):
        RareDiseaseFileLibrary(paper_client, llm_client).get_papers(query)


def test_rare_disease_get_papers(mock_paper_client: Any, mock_llm_client: Any, json_load: Any) -> None:
    rare_disease_paper = Paper(**json_load("rare_disease_paper.json"))
    other_paper = Paper(**json_load("other_paper.json"))
    ids = [rare_disease_paper.props["pmid"], other_paper.props["pmid"]]
    paper_client = mock_paper_client(ids, rare_disease_paper, other_paper)
    allowed_categories = ["genetic disease", "other"]
    llm_client = mock_llm_client(
        "genetic disease",
        "other",
        "other",
        "other",
    )
    query = {"gene_symbol": "gene"}
    result = RareDiseaseFileLibrary(paper_client, llm_client, allowed_categories).get_papers(query)
    assert paper_client.last_call("search") == ({"query": "gene pubmed pmc open access[filter]"},)
    assert paper_client.last_call("fetch") == (other_paper.props["pmid"], {"include_fulltext": True})
    assert llm_client.last_call("prompt_file")[2]["params"]["abstract"] == "We report on ..."
    assert result and len(result) == 2
    assert result[0] == rare_disease_paper

    # Remove all paper_finding_few_shot_*.txt from lib/evagg/content/prompts
    for file in os.listdir("lib/evagg/content/prompts"):
        if file.startswith("paper_finding_few_shot_"):
            os.remove(os.path.join("lib/evagg/content/prompts", file))


async def test_rare_disease_get_all_papers(mock_paper_client: Any, mock_llm_client: Any, json_load: Any) -> None:
    rare_disease_paper = Paper(**json_load("rare_disease_paper.json"))
    other_paper = Paper(**json_load("other_paper.json"))
    ids = [rare_disease_paper.props["pmid"], other_paper.props["pmid"]]
    paper_client = mock_paper_client(ids, rare_disease_paper, other_paper)
    llm_client = mock_llm_client(
        "genetic disease",
        "genetic disease",
        "genetic disease",
        "other",
        "other",
        "other",
    )
    query = {"gene_symbol": "gene"}
    result = await RareDiseaseFileLibrary(paper_client, llm_client)._get_all_papers(query)
    assert paper_client.last_call("search") == ({"query": "gene pubmed pmc open access[filter]"},)
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
        "citation": paper.props["citation"],
        "abstract": paper.props["abstract"],
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


def test_paper_list_library_init(mock_paper_client: Any) -> None:
    """Test PaperListLibrary initialization."""
    paper_client = mock_paper_client()
    gene_pmid_mapping = {"FBN2": ["26084686", "32184806"], "EXOC2": ["32639540"]}
    
    library = PaperListLibrary(paper_client, gene_pmid_mapping)
    
    assert library._paper_client == paper_client
    assert library._gene_pmid_mapping == gene_pmid_mapping


def test_paper_list_library_get_papers(mock_paper_client: Any, json_load: Any) -> None:
    """Test PaperListLibrary.get_papers method."""
    # Create mock papers
    paper1 = Paper(**json_load("rare_disease_paper.json"))  
    paper2 = Paper(**json_load("other_paper.json"))
    
    # Update IDs to match our test PMIDs
    paper1.id = "26084686"
    paper1.props["pmid"] = "26084686"
    paper2.id = "32184806"
    paper2.props["pmid"] = "32184806"
    
    # Set up mock client to return the papers in order
    paper_client = mock_paper_client(paper1, paper2)
    
    # Create library with gene-PMID mapping
    gene_pmid_mapping = {
        "FBN2": ["26084686", "32184806"],
        "EXOC2": ["32639540"]  # This PMID won't be tested here
    }
    library = PaperListLibrary(paper_client, gene_pmid_mapping)
    
    # Test with FBN2 gene
    query = {"gene_symbol": "FBN2"}
    result = library.get_papers(query)
    
    # Should return both papers for FBN2
    assert len(result) == 2
    assert paper1 in result
    assert paper2 in result
    
    # Verify the correct calls were made
    assert paper_client.call_count("fetch") == 2
    assert paper_client.last_call("fetch") == ("32184806", {"include_fulltext": True})


def test_paper_list_library_unknown_gene(mock_paper_client: Any) -> None:
    """Test PaperListLibrary with unknown gene symbol."""
    paper_client = mock_paper_client()  # No responses needed since no fetch calls expected
    gene_pmid_mapping = {"FBN2": ["26084686", "32184806"]}
    
    library = PaperListLibrary(paper_client, gene_pmid_mapping)
    
    # Test with unknown gene
    query = {"gene_symbol": "UNKNOWN_GENE"}
    result = library.get_papers(query)
    
    # Should return empty list
    assert len(result) == 0
    assert result == []
    
    # Fetch should not be called
    assert paper_client.call_count("fetch") == 0


def test_paper_list_library_no_gene_symbol(mock_paper_client: Any) -> None:
    """Test PaperListLibrary raises error when gene_symbol not provided."""
    paper_client = mock_paper_client()
    gene_pmid_mapping = {"FBN2": ["26084686"]}
    
    library = PaperListLibrary(paper_client, gene_pmid_mapping)
    
    # Test with query missing gene_symbol
    query = {"some_other_key": "value"}
    
    with pytest.raises(ValueError, match="Minimum requirement to search is to input a gene symbol."):
        library.get_papers(query)


def test_paper_list_library_failed_fetch(mock_paper_client: Any) -> None:
    """Test PaperListLibrary handles failed paper fetches gracefully."""
    # Set up mock to return None (fetch failure)
    paper_client = mock_paper_client(None)
    
    gene_pmid_mapping = {"FBN2": ["invalid_pmid"]}
    library = PaperListLibrary(paper_client, gene_pmid_mapping)
    
    query = {"gene_symbol": "FBN2"}
    result = library.get_papers(query)
    
    # Should return empty list when fetch fails
    assert len(result) == 0
    assert result == []
    
    # Verify fetch was called once
    assert paper_client.call_count("fetch") == 1
    assert paper_client.last_call("fetch") == ("invalid_pmid", {"include_fulltext": True})


def test_caching(mock_paper_client: Any, mock_llm_client: Any, json_load: Any) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:

        # Mock get_run_path to return the temporary directory.
        with patch("lib.evagg.utils.cache.get_run_path", return_value=tmpdir):

            # verify no cache exists.
            assert not os.path.exists(
                os.path.join(tmpdir, "results_cache", "RareDiseaseFileLibrary", "get_papers_gene.json")
            )

            rare_disease_paper = Paper(**json_load("rare_disease_paper.json"))
            paper_client = mock_paper_client([rare_disease_paper.props["pmid"]], rare_disease_paper)
            llm_client = mock_llm_client("genetic disease")
            query = {"gene_symbol": "gene"}
            library = RareDiseaseLibraryCached(
                paper_client=paper_client, llm_client=llm_client, use_previous_cache=False
            )
            result = library.get_papers(query)

            assert len(result) == 1
            assert result[0] == rare_disease_paper
            # verify cache was created.
            assert os.path.exists(
                os.path.join(tmpdir, "results_cache", "RareDiseaseFileLibrary", "get_papers_gene.json")
            )

            # The injected dependencies will be exhausted, so if we don't use the cache, we'll get an error.
            result = library.get_papers(query)
            assert len(result) == 1
            assert result[0] == rare_disease_paper
