import pytest

from lib.evagg.ref import NCBIGeneReference


@pytest.fixture
def single_symbol_direct_match():
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
def single_symbol_synonym_match():
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
def single_symbol_miss():
    return {}


@pytest.fixture
def multi_symbol():
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


def test_single_id_for_symbol_direct(mocker, single_symbol_direct_match):
    mocker.patch("lib.evagg.ref._ncbi.NCBIGeneReference._get_json", return_value=single_symbol_direct_match)

    result = NCBIGeneReference.gene_id_for_symbol("FAM111B")
    assert result == {"FAM111B": 374393}


def test_single_id_for_symbol_synonym(mocker, single_symbol_synonym_match):
    mocker.patch("lib.evagg.ref._ncbi.NCBIGeneReference._get_json", return_value=single_symbol_synonym_match)

    result = NCBIGeneReference.gene_id_for_symbol("FEB11")
    assert result == {}

    result = NCBIGeneReference.gene_id_for_symbol("FEB11", allow_synonyms=True)
    assert result == {"FEB11": 57094}

    # Verify that this would also work for the direct match.
    result = NCBIGeneReference.gene_id_for_symbol("CPA6")
    assert result == {"CPA6": 57094}


def test_single_id_for_symbol_miss(mocker, single_symbol_miss, single_symbol_direct_match):
    mocker.patch("lib.evagg.ref._ncbi.NCBIGeneReference._get_json", return_value=single_symbol_miss)

    result = NCBIGeneReference.gene_id_for_symbol("FAM111B")
    assert result == {}

    mocker.patch("lib.evagg.ref._ncbi.NCBIGeneReference._get_json", return_value=single_symbol_direct_match)

    result = NCBIGeneReference.gene_id_for_symbol("not a gene")
    assert result == {}


def test_multi_id_for_symbol_match(mocker, multi_symbol):
    mocker.patch("lib.evagg.ref._ncbi.NCBIGeneReference._get_json", return_value=multi_symbol)

    result = NCBIGeneReference.gene_id_for_symbol(["FAM111B", "FEB11"])
    assert result == {"FAM111B": 374393}

    result = NCBIGeneReference.gene_id_for_symbol(["FAM111B", "FEB11"], allow_synonyms=True)
    assert result == {"FAM111B": 374393, "FEB11": 57094}

    result = NCBIGeneReference.gene_id_for_symbol(["FAM111B", "FEB11", "not a gene"], allow_synonyms=True)
    assert result == {"FAM111B": 374393, "FEB11": 57094}


@pytest.fixture
def single_id_match():
    return {
        "reports": [
            {
                "gene": {
                    "gene_id": "374393",
                    "symbol": "FAM111B",
                },
                "query": ["374393"],
            }
        ],
        "total_count": 1,
    }


@pytest.fixture
def single_id_miss():
    return {}


@pytest.fixture
def multi_id_match():
    return {
        "reports": [
            {
                "gene": {
                    "gene_id": "374393",
                    "symbol": "FAM111B",
                },
                "query": ["374393"],
            },
            {
                "gene": {
                    "gene_id": "57094",
                    "symbol": "FEB11",
                },
                "query": ["57094"],
            },
        ],
        "total_count": 2,
    }


def test_single_symbol_for_id_match(mocker, single_id_match):
    mocker.patch("lib.evagg.ref._ncbi.NCBIGeneReference._get_json", return_value=single_id_match)

    result = NCBIGeneReference.gene_symbol_for_id([374393])
    assert result == {374393: "FAM111B"}


def test_single_symbol_for_id_miss(mocker, single_id_miss, single_id_match):
    mocker.patch("lib.evagg.ref._ncbi.NCBIGeneReference._get_json", return_value=single_id_miss)

    result = NCBIGeneReference.gene_symbol_for_id([374393])
    assert result == {}

    mocker.patch("lib.evagg.ref._ncbi.NCBIGeneReference._get_json", return_value=single_id_match)

    result = NCBIGeneReference.gene_symbol_for_id([-1])
    assert result == {}


def test_multi_symbol_for_id_match(mocker, multi_id_match):
    mocker.patch("lib.evagg.ref._ncbi.NCBIGeneReference._get_json", return_value=multi_id_match)

    result = NCBIGeneReference.gene_symbol_for_id([374393, 57094])
    assert result == {374393: "FAM111B", 57094: "FEB11"}
