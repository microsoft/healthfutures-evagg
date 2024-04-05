import pytest

from lib.evagg.ref import HPOReference


def test_compare() -> None:
    term_near1 = "HP:0000123"  # Nephritis
    term_near2 = "HP:0000124"  # Renal tubular dysfunction
    term_far = "HP:0001166"  # Arachnodactyly

    reference = HPOReference()

    assert reference.compare(term_near1, term_near2) > reference.compare(term_near1, term_far)
    assert reference.compare(term_near1, term_near2) > reference.compare(term_near1, term_far)


def test_compare_set() -> None:
    term_near1 = "HP:0000123"  # Nephritis
    term_near2 = "HP:0000124"  # Renal tubular dysfunction
    term_far = "HP:0001166"  # Arachnodactyly

    reference = HPOReference()

    result = reference.compare_set([term_near1, term_near2], [term_far])
    assert term_near1 in result and result[term_near1][1] == term_far
    assert term_near2 in result and result[term_near2][1] == term_far

    result = reference.compare_set([term_near1], [term_near2, term_far])
    assert term_near1 in result and result[term_near1][1] == term_near2


def test_translate() -> None:
    # Method has not been implemented.
    with pytest.raises(NotImplementedError):
        HPOReference().translate("Nephritis")
