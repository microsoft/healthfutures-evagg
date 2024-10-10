from typing import Any

import pytest

from lib.evagg.content.variant import HGVSVariantComparator, HGVSVariantFactory
from lib.evagg.ref import INormalizeVariants, IRefSeqLookupClient, IValidateVariants, IVariantLookupClient
from lib.evagg.types import HGVSVariant


def test_compare() -> None:
    comparator = HGVSVariantComparator()

    # Test two equivalent variants
    v1 = HGVSVariant("c.123A>G", "COQ2", None, True, True, None, None, [])
    assert comparator.compare(v1, v1) is v1
    v1 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.7)", True, True, None, None, [])
    assert comparator.compare(v1, v1) is v1

    # Test different variants
    v1 = HGVSVariant("c.123A>G", "COQ2", None, True, True, None, None, [])
    v2 = HGVSVariant("c.321A>G", "COQ2", None, True, True, None, None, [])
    assert comparator.compare(v1, v2) is None
    v1 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.7)", True, True, None, None, [])
    v2 = HGVSVariant("c.321A>G", "COQ2", "NP_123456.1(NM_123456.7)", True, True, None, None, [])
    assert comparator.compare(v1, v2) is None

    # Test two variants with different refseqs.
    v1 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.7)", True, True, None, None, [])
    v2 = HGVSVariant("c.123A>G", "COQ2", "NM_123456.8", True, True, None, None, [])
    assert comparator.compare(v1, v2) is v1
    v2 = HGVSVariant("c.123A>G", "COQ2", "NM_654321.8", True, True, None, None, [])
    assert comparator.compare(v1, v2) is None
    assert comparator.compare(v1, v2, True) is v1

    # Test variants with different completeness.
    v1 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.7)", True, True, None, None, [])
    v2 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.7)", False, True, None, None, [])
    assert comparator.compare(v1, v2) is v1
    v3 = HGVSVariant("c.123A>G", "COQ2", "NM_123456.7", False, True, None, None, [])
    assert comparator.compare(v2, v3) is v2
    assert comparator.compare(v1, v3) is v3
    v4 = HGVSVariant("c.123A>G", "COQ2", None, False, True, None, None, [])
    assert comparator.compare(v1, v4) is None
    # v4 has no refseq, so comparison will fail unless we disregard refseqs.
    assert comparator.compare(v1, v4, True) is v1

    # Test variants with equal completeness but different refseq version numbers.
    v1 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1", True, True, None, None, [])
    v2 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.2", True, True, None, None, [])
    assert comparator.compare(v1, v2) is v2
    v2 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.0", True, True, None, None, [])
    assert comparator.compare(v1, v2) is v1

    v1 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.7)", True, True, None, None, [])
    v2 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.6)", True, True, None, None, [])
    assert comparator.compare(v1, v2) is v1
    v2 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.8)", True, True, None, None, [])
    assert comparator.compare(v1, v2) is v2

    # Test weird edge cases.
    # V1 as a fallback because at least one refseq is shared, but there are not different
    # version numbers for the shared refseqs
    v2 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_654321.8)", True, True, None, None, [])
    assert comparator.compare(v1, v2) is v1


def test_compare_via_protein_consequence() -> None:
    v2 = HGVSVariant("p.Ala123Gly", "COQ2", "NP_123456.1", True, True, None, None, [])
    v1 = HGVSVariant("c.123A>G", "COQ2", "NM_123456.7", True, True, None, v2, [])
    comparator = HGVSVariantComparator()
    assert comparator.compare(v1, v2) is v1
    assert comparator.compare(v2, v1) is v1


def test_compare_via_coding_equivalent() -> None:
    pvar = HGVSVariant("p.Arg437Ter", "EXOC2", "NP_060773.3", True, True, None, None, [])
    cvar = HGVSVariant("c.1309C>T", "EXOC2", "NM_018303.6", True, True, None, pvar, [])
    gvar = HGVSVariant("g.576766G>A", "EXOC2", "NC_000006.11", True, True, None, None, [cvar])
    comparator = HGVSVariantComparator()
    assert comparator.compare(cvar, pvar) is cvar
    assert comparator.compare(pvar, cvar) is cvar
    assert comparator.compare(cvar, gvar) is cvar
    assert comparator.compare(gvar, cvar) is cvar
    assert comparator.compare(pvar, gvar) is pvar
    assert comparator.compare(gvar, pvar) is pvar


def test_consolidate() -> None:
    pvar = HGVSVariant("p.Arg437Ter", "EXOC2", "NP_060773.3", True, True, None, None, [])
    cvar = HGVSVariant("c.1309C>T", "EXOC2", "NM_018303.6", True, True, None, pvar, [])
    gvar = HGVSVariant("g.576766G>A", "EXOC2", "NC_000006.11", True, True, None, None, [cvar])

    comparator = HGVSVariantComparator()
    result = comparator.consolidate([pvar, cvar, gvar])

    assert len(result) == 1
    assert cvar in result
    assert pvar in result[cvar]
    assert cvar in result[cvar]
    assert gvar in result[cvar]


@pytest.fixture
def mock_validator(mock_client: Any) -> Any:
    return mock_client(IValidateVariants)


@pytest.fixture
def mock_normalizer(mock_client: Any) -> Any:
    return mock_client(INormalizeVariants)


@pytest.fixture
def mock_lookup_client(mock_client: Any) -> Any:
    return mock_client(IVariantLookupClient)


@pytest.fixture
def mock_refseq_client(mock_client: Any) -> Any:
    return mock_client(IRefSeqLookupClient)


def test_factory_parse_c_dot(mock_validator, mock_normalizer, mock_lookup_client, mock_refseq_client) -> None:

    def standard_normalizer() -> INormalizeVariants:
        return mock_normalizer(
            {
                "normalized_description": "NM_000059.4:c.1114A>C",
                "protein": {"description": "NM_000059.4(NP_000050.3):p.(Asn372His)"},
            },
            {
                "normalized_description": "NP_000050.3:p.Asn372His",
                "equivalent_descriptions": {"p": [{"description": "NP_000050.3:p.N372H"}]},
            },
        )

    # Parse a valid c. description
    validator = mock_validator((True, None))
    factory = HGVSVariantFactory(validator, standard_normalizer(), mock_lookup_client({}), mock_refseq_client(None))
    result = factory.parse("c.1114A>C", "BRCA2", "NM_000059.4")

    assert result.hgvs_desc == "c.1114A>C"
    assert result.gene_symbol == "BRCA2"
    assert result.refseq == "NM_000059.4"
    assert result.refseq_predicted is False
    assert result.valid is True
    assert result.validation_error is None
    assert result.protein_consequence is not None
    assert result.protein_consequence.hgvs_desc == "p.Asn372His"
    assert result.protein_consequence.refseq == "NP_000050.3"
    assert result.protein_consequence.refseq_predicted is False
    assert result.protein_consequence.valid is True
    assert result.protein_consequence.validation_error is None
    assert result.protein_consequence.protein_consequence is None
    assert result.protein_consequence.coding_equivalents == []
    assert result.coding_equivalents == []

    # Parse with an incomplete refseq that can be autocompleted
    validator = mock_validator((True, None))
    refseq_client = mock_refseq_client("NM_000059.4")
    factory = HGVSVariantFactory(validator, standard_normalizer(), mock_lookup_client({}), refseq_client)
    result = factory.parse("c.1114A>C", "BRCA2", "NM_000059")

    assert result.hgvs_desc == "c.1114A>C"
    assert result.gene_symbol == "BRCA2"
    assert result.refseq == "NM_000059.4"
    assert result.refseq_predicted is True
    assert result.valid is True
    assert result.validation_error is None
    assert result.protein_consequence is not None
    assert result.protein_consequence.hgvs_desc == "p.Asn372His"
    assert result.protein_consequence.refseq == "NP_000050.3"
    assert result.protein_consequence.refseq_predicted is True
    assert result.protein_consequence.valid is True
    assert result.protein_consequence.validation_error is None
    assert result.protein_consequence.protein_consequence is None
    assert result.protein_consequence.coding_equivalents == []
    assert result.coding_equivalents == []

    # Parse with an incomplete refseq that can't be autocompleted
    validator = mock_validator((False, None))
    refseq_client = mock_refseq_client(None)
    factory = HGVSVariantFactory(validator, mock_normalizer({}), mock_lookup_client({}), refseq_client)
    result = factory.parse("c.1114A>C", "BRCA2", "NM_BOGUS")

    assert result.hgvs_desc == "c.1114A>C"
    assert result.gene_symbol == "BRCA2"
    assert result.refseq == "NM_BOGUS"
    assert result.refseq_predicted is False
    assert result.valid is False
    assert result.validation_error is None
    assert result.protein_consequence is None
    assert result.coding_equivalents == []

    # Parse with a missing refseq
    validator = mock_validator((True, None))
    refseq_client = mock_refseq_client("NM_000059.4")
    factory = HGVSVariantFactory(validator, standard_normalizer(), mock_lookup_client({}), refseq_client)
    result = factory.parse("c.1114A>C", "BRCA2", refseq=None)

    assert result.hgvs_desc == "c.1114A>C"
    assert result.gene_symbol == "BRCA2"
    assert result.refseq == "NM_000059.4"
    assert result.refseq_predicted is True
    assert result.valid is True
    assert result.validation_error is None
    assert result.protein_consequence is not None
    assert result.protein_consequence.hgvs_desc == "p.Asn372His"
    assert result.protein_consequence.refseq == "NP_000050.3"
    assert result.protein_consequence.refseq_predicted is True
    assert result.protein_consequence.valid is True
    assert result.protein_consequence.validation_error is None
    assert result.protein_consequence.protein_consequence is None
    assert result.protein_consequence.coding_equivalents == []
    assert result.coding_equivalents == []

    # Parse with no gene symbol when needed
    with pytest.raises(ValueError):
        factory = HGVSVariantFactory(
            mock_validator(()), mock_normalizer({}), mock_lookup_client({}), mock_refseq_client(None)
        )
        factory.parse("c.1114A>C", None, "NM_000059.4")

    # Parse with failed validation
    validator = mock_validator((False, "ESEQUENCEMISMATCH"))
    factory = HGVSVariantFactory(validator, standard_normalizer(), mock_lookup_client({}), mock_refseq_client(None))
    result = factory.parse("c.1114G>C", "BRCA2", "NM_000059.4")  # Incorrect reference base

    assert result.hgvs_desc == "c.1114G>C"
    assert result.gene_symbol == "BRCA2"
    assert result.refseq == "NM_000059.4"
    assert result.refseq_predicted is False
    assert result.valid is False
    assert result.validation_error == "ESEQUENCEMISMATCH"
    assert result.protein_consequence is None
    assert result.coding_equivalents == []

    # Parse with intronic variant (no protein consequence)
    validator = mock_validator((False, "EINTRONIC"))
    normalizer = mock_normalizer({"error_message": "EINTRONIC"})
    factory = HGVSVariantFactory(validator, normalizer, mock_lookup_client({}), mock_refseq_client(None))
    result = factory.parse("c.8332-1591T>G", "BRCA2", "NC_000013.11(NM_000059.4)")

    assert result.hgvs_desc == "c.8332-1591T>G"
    assert result.gene_symbol == "BRCA2"
    assert result.refseq == "NC_000013.11(NM_000059.4)"
    assert result.refseq_predicted is False
    assert result.valid is False
    assert result.validation_error == "EINTRONIC"
    assert result.protein_consequence is None
    assert result.coding_equivalents == []

    # Parse intronic variant with incomplete refseq (missing genomic)
    validator = mock_validator((False, "EINTRONIC"))
    normalizer = mock_normalizer({"error_message": "EINTRONIC"})
    refseq_client = mock_refseq_client("NC_000013.11")
    factory = HGVSVariantFactory(validator, normalizer, mock_lookup_client({}), refseq_client)
    result = factory.parse("c.8332-1591T>G", "BRCA2", "NM_000059.4")

    assert result.hgvs_desc == "c.8332-1591T>G"
    assert result.gene_symbol == "BRCA2"
    assert result.refseq == "NC_000013.11(NM_000059.4)"
    assert result.refseq_predicted is False
    assert result.valid is False
    assert result.validation_error == "EINTRONIC"
    assert result.protein_consequence is None
    assert result.coding_equivalents == []

    # Parse unsupported variant with valid refseq
    validator = mock_validator((False, "ESYNTAXUC"))
    normalizer = mock_normalizer({"error_message": "ESYNTAXUC"})
    factory = HGVSVariantFactory(validator, normalizer, mock_lookup_client({}), mock_refseq_client(None))
    result = factory.parse("BOGUS", "BRCA2", "NM_000059.4")

    assert result.hgvs_desc == "BOGUS"
    assert result.gene_symbol == "BRCA2"
    assert result.refseq == "NM_000059.4"
    assert result.refseq_predicted is False
    assert result.valid is False
    assert result.validation_error == "ESYNTAXUC"
    assert result.protein_consequence is None
    assert result.coding_equivalents == []

    # Parse unsupported variant with missing refseq
    with pytest.raises(ValueError):
        factory.parse("BOGUS", "BRCA2", refseq=None)


def test_factory_parse_p_dot(mock_validator, mock_normalizer, mock_lookup_client, mock_refseq_client) -> None:
    # Parse a valid p. description
    validator = mock_validator((True, None))
    normalizer = mock_normalizer(
        {
            "normalized_description": "NP_000050.3:p.Asn372His",
            "equivalent_descriptions": {"p": [{"description": "NP_000050.3:p.N372H"}]},
        }
    )
    factory = HGVSVariantFactory(validator, normalizer, mock_lookup_client({}), mock_refseq_client(None))
    result = factory.parse("p.Asn372His", "BRCA2", "NP_000050.3")

    assert result.hgvs_desc == "p.Asn372His"
    assert result.gene_symbol == "BRCA2"
    assert result.refseq == "NP_000050.3"
    assert result.refseq_predicted is False
    assert result.valid is True
    assert result.validation_error is None
    assert result.protein_consequence is None
    assert result.coding_equivalents == []

    # Parse with a p. description that changes when normalized
    validator = mock_validator((True, None))
    normalizer = mock_normalizer(
        {
            "normalized_description": "NP_000050.3:p.Asn372Ala",
            "equivalent_descriptions": {"p": [{"description": "NP_000050.3:p.N372A"}]},
        }
    )
    factory = HGVSVariantFactory(validator, normalizer, mock_lookup_client({}), mock_refseq_client(None))
    result = factory.parse("p.N372A", "BRCA2", "NP_000050.3")

    assert result.hgvs_desc == "p.Asn372Ala"
    assert result.gene_symbol == "BRCA2"
    assert result.refseq == "NP_000050.3"
    assert result.refseq_predicted is False
    assert result.valid is True
    assert result.validation_error is None
    assert result.protein_consequence is None
    assert result.coding_equivalents == []

    # Parse with a predicted p. description
    validator = mock_validator((True, None))
    normalizer = mock_normalizer(
        {
            "normalized_description": "NP_000050.3:p.(Asn372Ala)",
            "equivalent_descriptions": {"p": [{"description": "NP_000050.3:p.(N372A)"}]},
        }
    )
    factory = HGVSVariantFactory(validator, normalizer, mock_lookup_client({}), mock_refseq_client(None))
    result = factory.parse("p.(N372A)", "BRCA2", "NP_000050.3")

    assert result.hgvs_desc == "p.Asn372Ala"
    assert result.gene_symbol == "BRCA2"
    assert result.refseq == "NP_000050.3"
    assert result.refseq_predicted is False
    assert result.valid is True
    assert result.validation_error is None
    assert result.protein_consequence is None
    assert result.coding_equivalents == []

    # Parse with a missing refseq
    validator = mock_validator((True, None))
    normalizer = mock_normalizer({"normalized_description": "NP_000050.3:p.Asn372His"})
    refseq_client = mock_refseq_client("NP_000050.3", "NM_000059.4")
    factory = HGVSVariantFactory(validator, normalizer, mock_lookup_client({}), refseq_client)
    result = factory.parse("p.Asn372His", "BRCA2", refseq=None)

    assert result.hgvs_desc == "p.Asn372His"
    assert result.gene_symbol == "BRCA2"
    assert result.refseq == "NP_000050.3"
    assert result.refseq_predicted is True
    assert result.valid is True
    assert result.validation_error is None
    assert result.protein_consequence is None
    assert result.coding_equivalents == []

    # Parse with a missing refseq and no predicted transcript refseq
    validator = mock_validator((True, None))
    normalizer = mock_normalizer({"normalized_description": "NP_000050.3:p.Asn372His"})
    refseq_client = mock_refseq_client("NP_000050.3", None)
    factory = HGVSVariantFactory(validator, normalizer, mock_lookup_client({}), refseq_client)
    result = factory.parse("p.Asn372His", "BRCA2", refseq=None)

    assert result.hgvs_desc == "p.Asn372His"
    assert result.gene_symbol == "BRCA2"
    assert result.refseq == "NP_000050.3"
    assert result.refseq_predicted is True
    assert result.valid is True
    assert result.validation_error is None
    assert result.protein_consequence is None
    assert result.coding_equivalents == []

    # Parse with failed validation
    validator = mock_validator((False, "EAMINOACIDMISMATCH"))
    normalizer = mock_normalizer({"error_message": "EAMINOACIDMISMATCH"})
    factory = HGVSVariantFactory(validator, normalizer, mock_lookup_client({}), mock_refseq_client(None))
    result = factory.parse("p.His372His", "BRCA2", "NP_000050.3")

    assert result.hgvs_desc == "p.His372His"
    assert result.gene_symbol == "BRCA2"
    assert result.refseq == "NP_000050.3"
    assert result.refseq_predicted is False
    assert result.valid is False
    assert result.validation_error == "EAMINOACIDMISMATCH"
    assert result.protein_consequence is None
    assert result.coding_equivalents == []

    # Parse a frameshift variant
    validator = mock_validator((False, "Frameshift validation not supported"))
    normalizer = mock_normalizer({"normalized_description": "NP_000050.3:p.His372fs"})
    factory = HGVSVariantFactory(validator, normalizer, mock_lookup_client({}), mock_refseq_client(None))
    result = factory.parse("p.His372fs", "BRCA2", "NP_000050.3")

    assert result.hgvs_desc == "p.His372fs"
    assert result.gene_symbol == "BRCA2"
    assert result.refseq == "NP_000050.3"
    assert result.refseq_predicted is False
    assert result.valid is True
    assert result.validation_error is None
    assert result.protein_consequence is None
    assert result.coding_equivalents == []


def test_factory_parse_g_dot(mock_validator, mock_normalizer, mock_lookup_client, mock_refseq_client) -> None:
    # Parse a valid g. description
    validator = mock_validator((True, None))
    normalizer = mock_normalizer(
        {
            "normalized_description": "NC_000013.11:g.32332592A>C",
            "equivalent_descriptions": {
                "c": [
                    {
                        "description": "NC_000013.11(NM_000059.4):c.1114A>C",
                        "protein_prediction": "NC_000013.11(NP_000050.3):p.(Asn372His)",
                        "selector": {"id": "NM_000059.4"},
                        "tag": {"id": "NM_000059.4", "details": "MANE Select"},
                    }
                ]
            },
        },
        {
            "normalized_description": "NM_000059.4:c.1114A>C",
            "protein": {"description": "NM_000059.4(NP_000050.3):p.(Asn372His)"},
        },
        {
            "normalized_description": "NP_000050.3:p.Asn372His",
            "equivalent_descriptions": {"p": [{"description": "NP_000050.3:p.N372H"}]},
        },
    )
    validator = mock_validator((True, None), (True, None), (True, None))
    factory = HGVSVariantFactory(validator, normalizer, mock_lookup_client({}), mock_refseq_client(None))
    result = factory.parse("g.32332592A>C", "BRCA2", "NC_000013.11")

    assert result.hgvs_desc == "g.32332592A>C"
    assert result.gene_symbol == "BRCA2"
    assert result.refseq == "NC_000013.11"
    assert result.refseq_predicted is False
    assert result.valid is True
    assert result.validation_error is None
    assert result.protein_consequence is None
    assert result.coding_equivalents is not None
    assert result.coding_equivalents[0].hgvs_desc == "c.1114A>C"
    assert result.coding_equivalents[0].gene_symbol == "BRCA2"
    assert result.coding_equivalents[0].refseq == "NM_000059.4"
    assert result.coding_equivalents[0].refseq_predicted is False
    assert result.coding_equivalents[0].valid is True
    assert result.coding_equivalents[0].validation_error is None
    assert result.coding_equivalents[0].protein_consequence is not None
    assert result.coding_equivalents[0].protein_consequence.hgvs_desc == "p.Asn372His"
    assert result.coding_equivalents[0].protein_consequence.refseq == "NP_000050.3"
    assert result.coding_equivalents[0].protein_consequence.refseq_predicted is False
    assert result.coding_equivalents[0].protein_consequence.valid is True
    assert result.coding_equivalents[0].protein_consequence.validation_error is None
    assert result.coding_equivalents[0].protein_consequence.protein_consequence is None
    assert result.coding_equivalents[0].protein_consequence.coding_equivalents == []

    # Parse a g. description without a refseq
    with pytest.raises(ValueError):
        factory.parse("g.32332592A>C", "BRCA2", refseq=None)


def test_factory_parse_m_dot(mock_validator, mock_normalizer, mock_lookup_client, mock_refseq_client) -> None:
    # Parse a valid m. description
    validator = mock_validator((True, None))
    normalizer = mock_normalizer({"normalized_description": "NC_012920.1:m.123A>G"})
    factory = HGVSVariantFactory(validator, normalizer, mock_lookup_client({}), mock_refseq_client(None))
    result = factory.parse("m.123A>G", None, "NC_012920.1")

    assert result.hgvs_desc == "m.123A>G"
    assert result.gene_symbol is None
    assert result.refseq == "NC_012920.1"
    assert result.refseq_predicted is False
    assert result.valid is True
    assert result.validation_error is None
    assert result.protein_consequence is None
    assert result.coding_equivalents == []

    # Parse a valid m. description without a refseq
    validator = mock_validator((True, None))
    normalizer = mock_normalizer({"normalized_description": "NC_012920.1:m.123A>G"})
    factory = HGVSVariantFactory(validator, normalizer, mock_lookup_client({}), mock_refseq_client(None))
    result = factory.parse("m.123A>G", None, refseq=None)

    assert result.hgvs_desc == "m.123A>G"
    assert result.gene_symbol is None
    assert result.refseq == "NC_012920.1"
    assert result.refseq_predicted is True
    assert result.valid is True
    assert result.validation_error is None
    assert result.protein_consequence is None
    assert result.coding_equivalents == []


def test_factory_parse_rsid(mock_validator, mock_normalizer, mock_lookup_client, mock_refseq_client) -> None:
    # test a valid rsid
    # test an invalid rsid

    # Test a valid rsid that results in a non-frameshift variant.
    lookup_client = mock_lookup_client(
        {
            "rs144848": {
                "hgvs_p": "NP_000050.3:p.Asn372His",
                "hgvs_c": "NM_000059.4:c.1114A>C",
                "hgvs_g": "NC_000013.11:g.32332592A>C",
                "gene": "BRCA2",
            }
        }
    )
    normalizer = mock_normalizer(
        {
            "normalized_description": "NM_000059.4:c.1114A>C",
            "protein": {"description": "NM_000059.4(NP_000050.3):p.(Asn372His)"},
        },
        {
            "normalized_description": "NP_000050.3:p.Asn372His",
            "equivalent_descriptions": {"p": [{"description": "NP_000050.3:p.N372H"}]},
        },
    )
    validator = mock_validator((True, None))

    factory = HGVSVariantFactory(validator, normalizer, lookup_client, mock_refseq_client(None))
    result = factory.parse_rsid("rs144848")

    assert result.hgvs_desc == "c.1114A>C"
    assert result.gene_symbol == "BRCA2"
    assert result.refseq == "NM_000059.4"
    assert result.refseq_predicted is False
    assert result.valid is True
    assert result.validation_error is None
    assert result.protein_consequence is not None
    assert result.protein_consequence.hgvs_desc == "p.Asn372His"
    assert result.protein_consequence.refseq == "NP_000050.3"
    assert result.protein_consequence.refseq_predicted is False
    assert result.protein_consequence.valid is True
    assert result.protein_consequence.validation_error is None
    assert result.protein_consequence.protein_consequence is None
    assert result.protein_consequence.coding_equivalents == []
    assert result.coding_equivalents == []

    # Test a valid rsid that results in a frameshift variant.
    lookup_client = mock_lookup_client(
        {
            "rs11571658": {
                "hgvs_p": "NP_000050.3:p.Leu2092fs",
                "hgvs_c": "NM_000059.4:c.6275_6276del",
                "hgvs_g": "NC_000013.11:g.32340630_32340631del",
                "gene": "BRCA2",
            }
        }
    )
    normalizer = mock_normalizer(
        {
            "normalized_description": "NM_000059.4:c.6275_6276del",
            "protein": {"description": "NM_000059.4(NP_000050.3):p.(Leu2092Profs*7)"},
        },
        {"normalized_description": "NP_000050.3:p.Leu2092fs"},
    )
    validator = mock_validator((True, None))

    factory = HGVSVariantFactory(validator, normalizer, lookup_client, mock_refseq_client(None))
    result = factory.parse_rsid("rs11571658")

    assert result.hgvs_desc == "c.6275_6276del"
    assert result.gene_symbol == "BRCA2"
    assert result.refseq == "NM_000059.4"
    assert result.refseq_predicted is False
    assert result.valid is True
    assert result.validation_error is None
    assert result.protein_consequence is not None
    assert result.protein_consequence.hgvs_desc == "p.Leu2092fs"
    assert result.protein_consequence.refseq == "NP_000050.3"
    assert result.protein_consequence.refseq_predicted is False
    assert result.protein_consequence.valid is True
    assert result.protein_consequence.validation_error is None
    assert result.protein_consequence.protein_consequence is None
    assert result.protein_consequence.coding_equivalents == []
    assert result.coding_equivalents == []

    # Test a valid rsid that only results in p.variant (unsure if this is a real case from dbSNP)
    lookup_client = mock_lookup_client(
        {
            "rs144848": {
                "hgvs_p": "NP_000050.3:p.Asn372His",
                "hgvs_g": "NC_000013.11:g.32332592A>C",
                "gene": "BRCA2",
            }
        }
    )
    normalizer = mock_normalizer(
        {
            "normalized_description": "NP_000050.3:p.Asn372His",
            "equivalent_descriptions": {"p": [{"description": "NP_000050.3:p.N372H"}]},
        },
    )
    validator = mock_validator((True, None))

    factory = HGVSVariantFactory(validator, normalizer, lookup_client, mock_refseq_client(None))
    result = factory.parse_rsid("rs144848")

    assert result.hgvs_desc == "p.Asn372His"
    assert result.gene_symbol == "BRCA2"
    assert result.refseq == "NP_000050.3"
    assert result.refseq_predicted is False
    assert result.valid is True
    assert result.validation_error is None
    assert result.protein_consequence is None
    assert result.coding_equivalents == []

    # Test a valid rsid that only results in a g.variant (unsure if this is a real case from dbSNP)
    lookup_client = mock_lookup_client(
        {
            "rs144848": {
                "hgvs_g": "NC_000013.11:g.32332592A>C",
                "gene": "BRCA2",
            }
        }
    )
    normalizer = mock_normalizer(
        {
            "normalized_description": "NC_000013.11:g.32332592A>C",
            "equivalent_descriptions": {
                "c": [
                    {
                        "description": "NC_000013.11(NM_000059.4):c.1114A>C",
                        "protein_prediction": "NC_000013.11(NP_000050.3):p.(Asn372His)",
                        "selector": {"id": "NM_000059.4"},
                        "tag": {"id": "NM_000059.4", "details": "MANE Select"},
                    }
                ]
            },
        },
        {
            "normalized_description": "NM_000059.4:c.1114A>C",
            "protein": {"description": "NM_000059.4(NP_000050.3):p.(Asn372His)"},
        },
        {
            "normalized_description": "NP_000050.3:p.Asn372His",
            "equivalent_descriptions": {"p": [{"description": "NP_000050.3:p.N372H"}]},
        },
    )
    validator = mock_validator((True, None), (True, None), (True, None))

    factory = HGVSVariantFactory(validator, normalizer, lookup_client, mock_refseq_client(None))
    result = factory.parse_rsid("rs144848")

    assert result.hgvs_desc == "g.32332592A>C"
    assert result.gene_symbol == "BRCA2"
    assert result.refseq == "NC_000013.11"
    assert result.refseq_predicted is False
    assert result.valid is True
    assert result.validation_error is None
    assert result.protein_consequence is None
    assert result.coding_equivalents is not None
    assert result.coding_equivalents[0].hgvs_desc == "c.1114A>C"
    assert result.coding_equivalents[0].gene_symbol == "BRCA2"
    assert result.coding_equivalents[0].refseq == "NM_000059.4"
    assert result.coding_equivalents[0].refseq_predicted is False
    assert result.coding_equivalents[0].valid is True
    assert result.coding_equivalents[0].validation_error is None
    assert result.coding_equivalents[0].protein_consequence is not None
    assert result.coding_equivalents[0].protein_consequence.hgvs_desc == "p.Asn372His"
    assert result.coding_equivalents[0].protein_consequence.refseq == "NP_000050.3"
    assert result.coding_equivalents[0].protein_consequence.refseq_predicted is False
    assert result.coding_equivalents[0].protein_consequence.valid is True
    assert result.coding_equivalents[0].protein_consequence.validation_error is None
    assert result.coding_equivalents[0].protein_consequence.protein_consequence is None
    assert result.coding_equivalents[0].protein_consequence.coding_equivalents == []

    # Test an invalid rsid.
    with pytest.raises(ValueError):
        factory = HGVSVariantFactory(
            mock_validator(()), mock_normalizer({}), mock_lookup_client({}), mock_refseq_client(None)
        )
        factory.parse_rsid("bogus")
