import json
import os
import tempfile
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from lib.evagg import PubMedFileLibrary, SimpleFileLibrary
from lib.evagg.types import Paper, Query
from lib.evagg.web.entrez import IEntrezClient

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


def return_paper1():
    # Generating paper
    paper1 = Paper(
        id="doi_1",
        citation="Test Paper 1",
        abstract="This is a test paper.",
        pmid="PMID123",
        pmcid="PMC123",
        is_pmc_oa="False",
    )

    return paper1


def return_find_ids_for_gene():
    return ["id_A"]


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
# Test if query is in the right format (gene:variant) and there are papers returned
# This is considered the normal functioning of the search method in the PubMedFileLibrary class.
def test_pubmedfilelibrary(mock_build_papers, mock_find_ids_for_gene, entrez_client):
    # TODO: If we can't call external resources, we need content from
    # somewhere. Thus, consider collections=[tmpdir]?

    paper1 = Paper(
        id="doi_1",
        citation="Test Paper 1",
        abstract="This is a test paper.",
        pmid="PMID123",
        pmcid="PMC123",
        is_pmc_oa="False",
    )

    library = PubMedFileLibrary(entrez_client, max_papers=1)

    # Isolating testing search method. No external resource calls.
    mock_query = MagicMock()
    mock_query.terms.return_value = ["gene_A"]  # Should be returned as the query term
    mock_find_ids_for_gene.return_value = ["id_A"]  # Replaces _find_ids_for_gene.
    mock_build_papers.return_value = paper1  # Replaces _build_papers.

    # Run search
    result = library.search(mock_query)

    # Assert statements
    mock_query.terms.assert_called_once()
    mock_find_ids_for_gene.assert_called_once_with(query="gene_A")
    mock_build_papers.assert_called_once_with(["id_A"])
    assert paper1 == result


# Test if an incorrect gene name is being passed in. If so, return an empty list.
def test_find_ids_for_gene_invalid_gene(mocker, entrez_client):
    # Mock the esearch method to return an XML string without an "IdList" element
    mock_esearch = mocker.patch("lib.evagg._library.IEntrezClient.esearch", return_value="<root></root>")

    # PubMedFileLibrary instance
    library = PubMedFileLibrary(entrez_client, max_papers=1)

    # Call the _find_ids_for_gene method with an invalid gene name
    result = library._find_ids_for_gene("invalid_gene")

    # Check that the result is an empty list
    assert result == []

    # Check that the esearch method was called with the correct arguments
    mock_esearch.assert_called_once_with(db="pmc", sort="relevance", retmax=1, retmode="xml", term="invalid_gene")


# Test if a valid gene name is being passed in yet there are no papers returned, to return an empty list.
def test_find_ids_for_gene_no_papers(mocker, entrez_client):
    # Mock the esearch method to return an XML string without any "Id" elements
    mock_esearch = mocker.patch(
        "lib.evagg._library.IEntrezClient.esearch", return_value="<root><IdList></IdList></root>"
    )

    # PubMedFileLibrary instance
    library = PubMedFileLibrary(entrez_client, max_papers=1)

    # Call the _find_ids_for_gene method with a valid gene name
    result = library._find_ids_for_gene("valid_gene")

    # Check that the result is an empty list
    assert result == []

    # Check that the esearch method was called with the correct arguments
    mock_esearch.assert_called_once_with(db="pmc", sort="relevance", retmax=1, retmode="xml", term="valid_gene")


# Test if query is not in the right format (gene:variant)
def test_search_invalid_query_format(entrez_client, mocker):
    # Create a PubMedFileLibrary instance
    library = PubMedFileLibrary(entrez_client, max_papers=1)

    # Mock the _find_ids_for_gene and _build_papers methods
    mocker.patch.object(library, "_find_ids_for_gene", return_value=[])
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

    # Mock the _find_ids_for_gene and _build_papers methods
    mocker.patch.object(library, "_find_ids_for_gene", return_value=[])
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

    # Mock the _find_ids_for_gene and _build_papers methods
    mocker.patch.object(library, "_find_ids_for_gene", return_value=[])
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


# NOTE that the following tests assume information of prival PubMedFileLibrary methods. This is not ideal, but does
# improve test coverage. I removed others and can certainly remove this.
# I placed this here for further discussion about the discussion around optimizing test coverage vs. testing public
# facing methods (e.g. search).


# Mock the esearch method to return a valid XML string
def test_find_ids_for_gene(mocker, entrez_client):
    byte_string = b"""<?xml version="1.0" encoding="UTF-8" ?>
    <!DOCTYPE eSearchResult PUBLIC "-//NLM//DTD esearch 20060628//EN" "https://eutils.ncbi.nlm.nih.gov/eutils/dtd/20060628/esearch.dtd">
    <eSearchResult><Count>48</Count><RetMax>1</RetMax><RetStart>0</RetStart><IdList><Id>8491582</Id></IdList><TranslationSet/><TranslationStack><TermSet><Term>RGSL1[All Fields]</Term><Field>All Fields</Field><Count>48</Count><Explode>N</Explode></TermSet><OP>GROUP</OP></TranslationStack><QueryTranslation>RGSL1[All Fields]</QueryTranslation></eSearchResult>"""
    string = byte_string.decode("utf-8")

    mock_esearch = mocker.patch("lib.evagg._library.IEntrezClient.esearch", return_value=string)

    # Create a PubMedFileLibrary instance
    library = PubMedFileLibrary(entrez_client, max_papers=1)

    # Call the _find_ids_for_gene method and check the result
    result = library._find_ids_for_gene("RGSL1")
    assert result == ["8491582"]

    # Check that the esearch method was called with the correct arguments
    mock_esearch.assert_called_once_with(db="pmc", sort="relevance", retmax=1, retmode="xml", term="RGSL1")


# Test the abstract and citation function and associated methods _generate_citation, and _extract_abstract.
# NOTE: I can write further tests for the functionality of the _generate_citation and _extract_abstract methods, but
# know this might not be appropriate given the discussion around testing private methods.
def test_get_abstract_and_citation():
    entrez_client = MagicMock()
    max_papers = 1
    library = PubMedFileLibrary(entrez_client, max_papers)

    # Mock the return value of the efetch method
    text_info = "Mock text info"
    entrez_client.efetch.return_value = text_info

    # Mock the return values of the _generate_citation and _extract_abstract methods
    citation = "Mock citation"
    doi = "Mock doi"
    pmcid_number = "Mock pmcid"
    abstract = "Mock abstract"
    library._generate_citation = MagicMock(return_value=(citation, doi, pmcid_number))
    library._extract_abstract = MagicMock(return_value=abstract)

    # Call the _get_abstract_and_citation method and check the result
    result = library._get_abstract_and_citation("12345")
    assert result == (citation, doi, abstract, pmcid_number)

    # Check that the efetch method was called with the correct arguments
    entrez_client.efetch.assert_called_once_with(db="pubmed", id="12345", retmode="text", rettype="abstract")

    # Check that the _generate_citation and _extract_abstract methods were called with the correct arguments
    library._generate_citation.assert_called_once_with(text_info)
    library._extract_abstract.assert_called_once_with(text_info)


# Checks that _generate_citation correctly extracts author's last name, year, journal abbreviation, DOI number, and
# PMCID from text_info string and constructs the citation correctly
# NOTE: this is likely not needed and could be handled further in _library.py
def test_generate_citation(entrez_client):
    max_papers = 1
    library = PubMedFileLibrary(entrez_client, max_papers)

    # Define a mock text_info string
    text_info = "1. J Neurochem. 2014 Mar;128(5):741-51. doi: 10.1111/jnc.12491. Epub 2013 Nov 13.\n\nSCA14 mutation V138E leads to partly unfolded PKCγ associated with an exposed \nC-terminus, altered kinetics, phosphorylation and enhanced insolubilization.\n\nJezierska J(1), Goedhart J, Kampinga HH, Reits EA, Verbeek DS.\n\nAuthor information:\n(1)Department of Genetics, University of Groningen, University Medical Center \nGroningen, Groningen, The Netherlands.\n\nThe protein kinase C γ (PKCγ) undergoes multistep activation and participates in \nvarious cellular processes in Purkinje cells. Perturbations in its \nphosphorylation state, conformation or localization can disrupt kinase \nsignalling, such as in spinocerebellar ataxia type 14 (SCA14) that is caused by \nmissense mutations in PRKCG encoding for PKCγ. We previously showed that SCA14 \nmutations enhance PKCγ membrane translocation upon stimulation owing to an \naltered protein conformation. As the faster translocation did not result in an \nincreased function, we examined how SCA14 mutations induce this altered \nconformation of PKCγ and what the consequences of this conformational change are \non PKCγ life cycle. Here, we show that SCA14-related PKCγ-V138E exhibits an \nexposed C-terminus as shown by fluorescence resonance energy \ntransfer-fluorescence lifetime imaging microscopy in living cells, indicative of \nits partial unfolding. This conformational change was associated with faster \nphorbol 12-myristate 13-acetate-induced translocation and accumulation of fully \nphosphorylated PKCγ in the insoluble fraction, which could be rescued by \ncoexpressing PDK1 kinase that normally triggers PKCγ autophosphorylation. We \npropose that the SCA14 mutation V138E causes unfolding of the C1B domain and \nexposure of the C-terminus of the PKCγ-V138E molecule, resulting in a decrease \nof functional kinase in the soluble fraction. Here, we show that the mutation \nV138E of the protein kinase C γ (PKCγ) C1B domain (PKCγ-V138E), which is \nimplicated in spinocerebellar ataxia type 14, exhibits a partially unfolded \nC-terminus. This leads to unusually fast phorbol 12-myristate 13-acetate-induced \nmembrane translocation and accumulation of phosphorylated PKCγ-V138E in the \ninsoluble fraction, causing loss of the functional kinase. In contrast to \ngeneral chaperones, coexpression of PKCγ's 'natural chaperone', PDK1 kinase, \ncould rescue the PKCγ-V138E phenotype.\n\n© 2013 International Society for Neurochemistry.\n\nDOI: 10.1111/jnc.12491\nPMID: 24134140 [Indexed for MEDLINE] "

    # Call the _generate_citation method and check the result
    result = library._generate_citation(text_info)
    expected_result = ("Jezierska (2014), J Neurochem., 10.1111/jnc.12491", "10.1111/jnc.12491", 0.0)
    assert result == expected_result
