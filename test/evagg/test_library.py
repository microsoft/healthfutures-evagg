import json
import os
import tempfile
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from lib.evagg import PubMedFileLibrary, SimpleFileLibrary
from lib.evagg.types import Paper, Query


def _paper_to_dict(paper: Paper) -> Dict[str, Any]:
    return {
        "id": paper.id,
        "evidence": paper.evidence,
        "citation": paper.citation,
        "abstract": paper.abstract,
        "props": paper.props,
    }


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


@patch.object(PubMedFileLibrary, "_find_ids_for_gene")
@patch.object(PubMedFileLibrary, "_build_papers")
def test_pubmedfilelibrary(mock_build_papers, mock_find_ids_for_gene):
    # TODO: If we can't call external resources, we need content from
    # somewhere. Thus, consider collections=[tmpdir]?
    # Generating paper
    paper1 = Paper(
        id="doi_1",
        citation="Test Paper 1",
        abstract="This is a test paper.",
        pmid="PMID123",
        pmcid="PMC123",
        is_pmc_oa="False"
    )

    # Call library class
    library = PubMedFileLibrary(email="ashleyconard@microsoft.com", max_papers=1)

    # Isolating testing search method. No external resource calls.
    mock_query = MagicMock() 
    mock_query.terms.return_value = ["gene_A"]
    mock_find_ids_for_gene.return_value = ["id_A"]  # Replaces _find_ids_for_gene.
    mock_build_papers.return_value = (paper1)  # Replaces _build_papers.

    # Run search
    result = library.search(mock_query)

    # Assert statements
    mock_query.terms.assert_called_once()
    mock_find_ids_for_gene.assert_called_once_with(query="gene_A")
    mock_build_papers.assert_called_once_with(["id_A"])
    assert paper1 == result
