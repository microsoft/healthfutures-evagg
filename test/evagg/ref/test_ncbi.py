import pytest

from lib.evagg.ref import NCBIGeneReference, NCBIVariantReference


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


@pytest.fixture
def valid_variant():
    return """
{
    "refsnp_id": "146010120",
    "primary_snapshot_data": {
        "placements_with_allele": [
            {
                "seq_id": "NM_001276.4",
                "is_ptlp": false,
                "placement_annot": {
                    "seq_type": "refseq_mrna",
                    "mol_type": "rna",
                    "seq_id_traits_by_assembly": [],
                    "is_aln_opposite_orientation": true,
                    "is_mismatch": false
                },
                "alleles": [
                    {
                        "allele": {
                            "spdi": {
                                "seq_id": "NM_001276.4",
                                "position": 184,
                                "deleted_sequence": "G",
                                "inserted_sequence": "G"
                            }
                        },
                        "hgvs": "NM_001276.4:c.104="
                    },
                    {
                        "allele": {
                            "spdi": {
                                "seq_id": "NM_001276.4",
                                "position": 184,
                                "deleted_sequence": "G",
                                "inserted_sequence": "A"
                            }
                        },
                        "hgvs": "NM_001276.4:c.104G>A"
                    }
                ]
            },
            {
                "seq_id": "NP_001267.2",
                "is_ptlp": false,
                "placement_annot": {
                    "seq_type": "refseq_prot",
                    "mol_type": "protein",
                    "seq_id_traits_by_assembly": [],
                    "is_aln_opposite_orientation": false,
                    "is_mismatch": false
                },
                "alleles": [
                    {
                        "allele": {
                            "spdi": {
                                "seq_id": "NP_001267.2",
                                "position": 34,
                                "deleted_sequence": "R",
                                "inserted_sequence": "R"
                            }
                        },
                        "hgvs": "NP_001267.2:p.Arg35="
                    },
                    {
                        "allele": {
                            "spdi": {
                                "seq_id": "NP_001267.2",
                                "position": 34,
                                "deleted_sequence": "R",
                                "inserted_sequence": "Q"
                            }
                        },
                        "hgvs": "NP_001267.2:p.Arg35Gln"
                    }
                ]
            }
        ]
    },
    "mane_select_ids": [
        "NM_001276.4"
    ]
}
"""


@pytest.fixture
def invalid_variant():
    return ""


def test_variant(mocker, valid_variant):
    mocker.patch("lib.evagg.ref._ncbi.NCBIVariantReference._entrez_fetch", return_value=valid_variant)
    result = NCBIVariantReference.hgvs_from_rsid("146010120")
    assert result == {"hgvsc": "NM_001276.4:c.104G>A", "hgvsp": "NP_001267.2:p.Arg35Gln"}

    result = NCBIVariantReference.hgvs_from_rsid("rs146010120")
    assert result == {"hgvsc": "NM_001276.4:c.104G>A", "hgvsp": "NP_001267.2:p.Arg35Gln"}


def test_missing_variant(mocker, invalid_variant):
    mocker.patch("lib.evagg.ref._ncbi.NCBIVariantReference._entrez_fetch", return_value=invalid_variant)
    result = NCBIVariantReference.hgvs_from_rsid("123456789")
    assert result == {}


def test_non_rsid():
    result = NCBIVariantReference.hgvs_from_rsid("not a rsid")
    assert result == {}
