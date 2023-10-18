import pytest

from lib.evagg.ref import NCBIGeneReference


@pytest.fixture
def single_gene_direct_match():
    return {
        "reports": [
            {
                "gene": {
                    "gene_id": "374393",
                    "symbol": "FAM111B",
                },
                "query": ["FAM111B"],
            }
        ],
        "total_count": 1,
    }


@pytest.fixture
def single_gene_indirect_match():
    return {
        "reports": [
            {
                "gene": {
                    "gene_id": "57094",
                    "symbol": "CPA6",
                },
                "query": ["FEB11"],
            }
        ],
        "total_count": 1,
    }


@pytest.fixture
def single_gene_miss():
    return {}


@pytest.fixture
def multi_gene():
    return {
        "reports": [
            {
                "gene": {
                    "gene_id": "374393",
                    "symbol": "FAM111B",
                },
                "query": ["FAM111B"],
            },
            {
                "gene": {
                    "gene_id": "57094",
                    "symbol": "CPA6",
                },
                "query": ["FEB11"],
            },
        ],
        "total_count": 2,
    }


def test_single_gene_direct(mocker, single_gene_direct_match):
    mocker.patch("lib.evagg.ref._ncbi.NCBIGeneReference._get_json", return_value=single_gene_direct_match)

    result = NCBIGeneReference.gene_id_for_symbol("FAM111B")
    assert result == {"FAM111B": 374393}


def test_single_gene_indirect(mocker, single_gene_indirect_match):
    mocker.patch("lib.evagg.ref._ncbi.NCBIGeneReference._get_json", return_value=single_gene_indirect_match)

    result = NCBIGeneReference.gene_id_for_symbol("FEB11")
    assert result == {}

    result = NCBIGeneReference.gene_id_for_symbol("FEB11", allow_synonyms=True)
    assert result == {"FEB11": 57094}

    # Verify that this would also work for the direct match.
    result = NCBIGeneReference.gene_id_for_symbol("CPA6")
    assert result == {"CPA6": 57094}


def test_single_gene_miss(mocker, single_gene_miss, single_gene_direct_match):
    mocker.patch("lib.evagg.ref._ncbi.NCBIGeneReference._get_json", return_value=single_gene_miss)

    result = NCBIGeneReference.gene_id_for_symbol("FAM11B")
    assert result == {}

    mocker.patch("lib.evagg.ref._ncbi.NCBIGeneReference._get_json", return_value=single_gene_direct_match)

    result = NCBIGeneReference.gene_id_for_symbol("not a gene")
    assert result == {}


def test_multi_gene(mocker, multi_gene):
    mocker.patch("lib.evagg.ref._ncbi.NCBIGeneReference._get_json", return_value=multi_gene)

    result = NCBIGeneReference.gene_id_for_symbol(["FAM111B", "FEB11"])
    assert result == {"FAM111B": 374393}

    result = NCBIGeneReference.gene_id_for_symbol(["FAM111B", "FEB11"], allow_synonyms=True)
    assert result == {"FAM111B": 374393, "FEB11": 57094}

    result = NCBIGeneReference.gene_id_for_symbol(["FAM111B", "FEB11", "not a gene"], allow_synonyms=True)
    assert result == {"FAM111B": 374393, "FEB11": 57094}
