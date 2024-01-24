import json
import os
import tempfile
import xml.etree.ElementTree as Et
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from lib.evagg import PubMedFileLibrary, SimpleFileLibrary
from lib.evagg.ref import IEntrezClient
from lib.evagg.types import Paper, Query

# TODO: focus on critical functions


def _paper_to_dict(paper: Paper) -> Dict[str, Any]:
    return {
        "id": paper.id,
        "evidence": paper.evidence,
        "citation": paper.citation,
        "abstract": paper.abstract,
        "props": paper.props,
    }


@pytest.fixture
def entrez_client():
    class DummyEntrezClient(IEntrezClient):
        def efetch(self, db: str, id: str, retmode: str | None, rettype: str | None) -> str:
            return ""

    return DummyEntrezClient()


def test_search():
    # Create a temporary directory and write some test papers to it
    with tempfile.TemporaryDirectory() as tmpdir:
        paper1 = Paper(id="1", citation="Test Paper 1", abstract="This is a test paper.", pmcid="PMC123")
        paper2 = Paper(id="2", citation="Test Paper 2", abstract="This is another test paper.", pmcid="PMC123")
        paper3 = Paper(id="3", citation="Test Paper 3", abstract="This is a third test paper.", pmcid="PMC123")
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


@patch.object(PubMedFileLibrary, "_find_pmids_for_gene")
@patch.object(PubMedFileLibrary, "_build_papers")
# Test if query is in the right format (gene:variant) and there are papers returned
# This is considered the normal functioning of the search method in the PubMedFileLibrary class.
def test_pubmedfilelibrary(mock_build_papers, mock_find_pmids_for_gene, entrez_client):
    # TODO: If we can't call external resources, we need content from
    # somewhere. Thus, consider collections=[tmpdir]?

    paper1 = Paper(
        id="doi_1",
        citation="Test Paper 1",
        abstract="This is a test paper.",
        pmid="PMID123",
        pmcid="PMC123",
        is_pmc_oa=False,
    )

    library = PubMedFileLibrary(entrez_client, max_papers=1)

    # Isolating testing search method. No external resource calls.
    mock_query = MagicMock()
    mock_query.terms.return_value = ["gene_A"]  # Should be returned as the query term
    mock_find_pmids_for_gene.return_value = ["id_A"]  # Replaces _find_pmids_for_gene.
    mock_build_papers.return_value = paper1  # Replaces _build_papers.

    # Run search
    result = library.search(mock_query)

    # Assert statements
    mock_find_pmids_for_gene.assert_called_once_with(query="gene_A")
    mock_build_papers.assert_called_once_with(["id_A"])
    assert paper1 == result


# Test if an incorrect gene name is being passed in. If so, return an empty list.
def test_find_pmids_for_gene_invalid_gene(mocker, entrez_client):
    # Mock the esearch method to return an XML string without an "IdList" element
    mock_esearch = mocker.patch("lib.evagg.library.IEntrezClient.esearch", return_value=Et.fromstring("<root></root>"))

    # PubMedFileLibrary instance
    library = PubMedFileLibrary(entrez_client, max_papers=1)

    # Call the _find_pmids_for_gene method with an invalid gene name
    result = library._find_pmids_for_gene("invalid_gene")

    # Check that the result is an empty list
    assert result == []

    # Check that the esearch method was called with the correct arguments
    mock_esearch.assert_called_once_with(db="pubmed", sort="relevance", retmax=1, term="invalid_gene", retmode="xml")


# Test if a valid gene name is being passed in yet there are no papers returned, to return an empty list.
def test_find_pmids_for_gene_no_papers(mocker, entrez_client):
    # Mock the esearch method to return an XML string without any "Id" elements
    mock_esearch = mocker.patch(
        "lib.evagg.library.IEntrezClient.esearch", return_value=Et.fromstring("<root><IdList></IdList></root>")
    )

    # PubMedFileLibrary instance
    library = PubMedFileLibrary(entrez_client, max_papers=1)

    # Call the _find_pmids_for_gene method with a valid gene name
    result = library._find_pmids_for_gene("valid_gene")

    # Check that the result is an empty list
    assert result == []

    # Check that the esearch method was called with the correct arguments
    mock_esearch.assert_called_once_with(db="pubmed", sort="relevance", retmax=1, term="valid_gene", retmode="xml")


# Test if query is not in the right format (gene:variant)
def test_search_invalid_query_format(entrez_client, mocker):
    # Create a PubMedFileLibrary instance
    library = PubMedFileLibrary(entrez_client, max_papers=1)

    # Mock the _find_pmids_for_gene and _build_papers methods
    mocker.patch.object(library, "_find_pmids_for_gene", return_value=[])
    mocker.patch.object(library, "_build_papers", return_value=set())

    # Create a mock IPaperQuery object with an invalid format
    mock_query = MagicMock()
    mock_query.terms.return_value = ["invalid_query"]

    # Call the search method with the mock query
    try:
        library.search(mock_query)
    except ValueError as e:
        assert str(e) == "Query must be in the format 'gene:variant'"


# Test if query is in the right format (gene:variant)
def test_search_valid_query_format(entrez_client, mocker):
    # Create a PubMedFileLibrary instance
    library = PubMedFileLibrary(entrez_client, max_papers=1)

    # Mock the _find_pmids_for_gene and _build_papers methods
    mocker.patch.object(library, "_find_pmids_for_gene", return_value=[])
    mocker.patch.object(library, "_build_papers", return_value=set())

    # Create a mock IPaperQuery object with a valid format
    mock_query = MagicMock()
    mock_query.terms.return_value = ["gene:variant"]

    # Call the search method with the mock query
    try:
        library.search(mock_query)
    except ValueError:
        raise AssertionError(
            "Search method raised ValueError for a valid query"
        )  # Checks that search method does not raise a ValueError with correct query. If search method does raise a
        # ValueError, this test will fail with the message here


# Test if query is in the right format (gene:variant) but there are no papers returned
def test_search_no_papers_returned(entrez_client, mocker):
    # Create a PubMedFileLibrary instance
    library = PubMedFileLibrary(entrez_client, max_papers=1)

    # Mock the _find_pmids_for_gene and _build_papers methods
    mocker.patch.object(library, "_find_pmids_for_gene", return_value=[])
    mocker.patch.object(library, "_build_papers", return_value=set())

    # Create a mock IPaperQuery object with a valid format
    mock_query = MagicMock()
    mock_query.terms.return_value = ["gene:variant"]

    # Call the search method with the mock query
    result = library.search(mock_query)

    # Check that the result is an empty set
    assert result == set(), "search method did not return an empty set for a query with no matching papers"


# Test the init method of the PubMedFileLibrary class
def test_init(entrez_client):
    max_papers = 5
    library = PubMedFileLibrary(entrez_client, max_papers)
    assert library._entrez_client == entrez_client
    assert library._max_papers == max_papers
