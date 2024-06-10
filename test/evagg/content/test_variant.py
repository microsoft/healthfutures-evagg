from lib.evagg.content.variant import HGVSVariantComparator
from lib.evagg.types import HGVSVariant


def test_compare() -> None:
    comparator = HGVSVariantComparator()

    # Test two equivalent variants
    v1 = HGVSVariant("c.123A>G", "COQ2", None, True, True, None, [])
    assert comparator.compare(v1, v1) is v1
    v1 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.7)", True, True, None, [])
    assert comparator.compare(v1, v1) is v1

    # Test different variants
    v1 = HGVSVariant("c.123A>G", "COQ2", None, True, True, None, [])
    v2 = HGVSVariant("c.321A>G", "COQ2", None, True, True, None, [])
    assert comparator.compare(v1, v2) is None
    v1 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.7)", True, True, None, [])
    v2 = HGVSVariant("c.321A>G", "COQ2", "NP_123456.1(NM_123456.7)", True, True, None, [])
    assert comparator.compare(v1, v2) is None

    # Test two variants with different refseqs.
    v1 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.7)", True, True, None, [])
    v2 = HGVSVariant("c.123A>G", "COQ2", "NM_123456.8", True, True, None, [])
    assert comparator.compare(v1, v2) is v1
    v2 = HGVSVariant("c.123A>G", "COQ2", "NM_654321.8", True, True, None, [])
    assert comparator.compare(v1, v2) is None
    assert comparator.compare(v1, v2, True) is v1

    # Test variants with different completeness.
    v1 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.7)", True, True, None, [])
    v2 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.7)", False, True, None, [])
    assert comparator.compare(v1, v2) is v1
    v3 = HGVSVariant("c.123A>G", "COQ2", "NM_123456.7", False, True, None, [])
    assert comparator.compare(v2, v3) is v2

    # Test variants with equal completeness but different refseq version numbers.
    v1 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1", True, True, None, [])
    v2 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.2", True, True, None, [])
    assert comparator.compare(v1, v2) is v2
    v2 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.0", True, True, None, [])
    assert comparator.compare(v1, v2) is v1

    v1 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.7)", True, True, None, [])
    v2 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.6)", True, True, None, [])
    assert comparator.compare(v1, v2) is v1
    v2 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.8)", True, True, None, [])
    assert comparator.compare(v1, v2) is v2

    # Test weird edge cases.
    # V1 as a fallback because at least one refseq is shared, but there are not different
    # version numbers for the shared refseqs
    v2 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_654321.8)", True, True, None, [])
    assert comparator.compare(v1, v2) is v1


def test_compare_via_protein_consequence() -> None:
    v2 = HGVSVariant("p.Ala123Gly", "COQ2", "NP_123456.1", True, True, None, [])
    v1 = HGVSVariant("c.123A>G", "COQ2", "NM_123456.7", True, True, v2, [])
    comparator = HGVSVariantComparator()
    assert comparator.compare(v1, v2) is v1
    assert comparator.compare(v2, v1) is v1


def test_compare_via_coding_equivalent() -> None:
    pvar = HGVSVariant("p.Arg437Ter", "EXOC2", "NP_060773.3", True, True, None, [])
    cvar = HGVSVariant("c.1309C>T", "EXOC2", "NM_018303.6", True, True, pvar, [])
    gvar = HGVSVariant("g.576766G>A", "EXOC2", "NC_000006.11", True, True, None, [cvar])
    comparator = HGVSVariantComparator()
    assert comparator.compare(cvar, pvar) is cvar
    assert comparator.compare(pvar, cvar) is cvar
    assert comparator.compare(cvar, gvar) is cvar
    assert comparator.compare(gvar, cvar) is cvar
    assert comparator.compare(pvar, gvar) is pvar
    assert comparator.compare(gvar, pvar) is pvar
