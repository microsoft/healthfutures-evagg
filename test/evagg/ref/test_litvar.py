import json
import time

import pytest

from lib.evagg.ref import LitVarReference


@pytest.fixture
def autocomplete_success():
    return """
[
    {
        "_id": "litvar@rs113488022##",
        "rsid": "rs113488022",
        "gene": [
            "BRAF"
        ],
        "name": "p.V600E",
        "hgvs": "p.V600E",
        "pmids_count": 21820,
        "flag_gene_variant": false,
        "flag_clingen_variant": false,
        "flag_rsid_variant": true,
        "data_clinical_significance": [
            "other",
            "drug-response",
            "pathogenic",
            "likely-pathogenic"
        ],
        "match": "Matched on hgvs <m>p.V600E</m>"
    }
]
"""


@pytest.fixture
def autocomplete_fail():
    return "[]"


@pytest.fixture
def variants_for_gene_success():
    # Ugly or not, this is representative of what the API responds.
    return "{'_id': 'litvar@rs9986821##', 'pmids_count': 1, 'rsid': 'rs9986821'}\n{'_id': 'litvar@rs998233805##', 'pmids_count': 1, 'rsid': 'rs998233805'}\n{'_id': 'litvar@rs993830683##', 'pmids_count': 2, 'rsid': 'rs993830683'}\n{'_id': 'litvar@rs9886143##', 'pmids_count': 2, 'rsid': 'rs9886143'}\n{'_id': 'litvar@rs986525858##', 'pmids_count': 1, 'rsid': 'rs986525858'}\n{'_id': 'litvar@rs986050##', 'pmids_count': 3, 'rsid': 'rs986050'}\n{'_id': 'litvar@rs978365655##', 'pmids_count': 1, 'rsid': 'rs978365655'}\n{'_id': 'litvar@rs964942##', 'pmids_count': 4, 'rsid': 'rs964942'}\n{'_id': 'litvar@rs9648716##', 'pmids_count': 15, 'rsid': 'rs9648716'}\n{'_id': 'litvar@rs9648715##', 'pmids_count': 1, 'rsid': 'rs9648715'}\n{'_id': 'litvar@rs9648696##', 'pmids_count': 28, 'rsid': 'rs9648696'}\n{'_id': 'litvar@rs964235659##', 'pmids_count': 2, 'rsid': 'rs964235659'}\n{'_id': 'litvar@rs9640168##', 'pmids_count': 4, 'rsid': 'rs9640168'}\n{'_id': 'litvar@rs960526891##', 'pmids_count': 1, 'rsid': 'rs960526891'}\n{'_id': 'litvar@rs956815092##', 'pmids_count': 2, 'rsid': 'rs956815092'}\n{'_id': 'litvar@rs956143558##', 'pmids_count': 3, 'rsid': 'rs956143558'}"  # noqa


@pytest.fixture
def variants_for_gene_fail():
    # Ugly or not, this is representative of what the API responds.
    return ""


@pytest.fixture
def pmids_for_variant_success():
    return """
{
    "pmids": [
        32768003,
        34865163,
        33161228,
        34209805,
        27656207
    ]
}
"""


@pytest.fixture
def pmids_for_variant_fail():
    return """
{
    "detail": "Variant not found: rs1234"
}
"""


def test_variant_autocomplete(mocker, autocomplete_success):
    mocker.patch("lib.evagg.ref.litvar.LitVarReference._requests_get_text", return_value=autocomplete_success)

    result = LitVarReference.variant_autocomplete("irrelevant search string")
    assert result == json.loads(autocomplete_success)


def test_variant_autocomplete_fail(mocker, autocomplete_fail):
    mocker.patch("lib.evagg.ref.litvar.LitVarReference._requests_get_text", return_value=autocomplete_fail)

    result = LitVarReference.variant_autocomplete("irrelevant search string")
    assert result == json.loads(autocomplete_fail)


def test_variants_for_gene(mocker, variants_for_gene_success):
    mocker.patch("lib.evagg.ref.litvar.LitVarReference._requests_get_text", return_value=variants_for_gene_success)

    print(f"##{variants_for_gene_success}##")
    result = LitVarReference.variants_for_gene("irrelevant gene symbol")

    for obj1, obj2 in zip(result, [json.loads(s) for s in variants_for_gene_success.replace("'", '"').split("\n")]):
        assert obj1 == obj2


def test_variants_for_gene_fail(mocker, variants_for_gene_fail):
    mocker.patch("lib.evagg.ref.litvar.LitVarReference._requests_get_text", return_value=variants_for_gene_fail)

    result = LitVarReference.variants_for_gene("irrelevant gene symbol")
    assert result == []


def test_pmids_for_variant(mocker, pmids_for_variant_success):
    mocker.patch("lib.evagg.ref.litvar.LitVarReference._requests_get_text", return_value=pmids_for_variant_success)

    result = LitVarReference.pmids_for_variant("litvar%40rs113488022%23%23")
    assert result == json.loads(pmids_for_variant_success)


def test_pmids_for_variant_fail(mocker, pmids_for_variant_success, pmids_for_variant_fail):
    mocker.patch("lib.evagg.ref.litvar.LitVarReference._requests_get_text", return_value=pmids_for_variant_success)

    # Variant validation fails.
    result = LitVarReference.pmids_for_variant("irrelevant variant id")
    assert result == {}

    # No variant provided
    result = LitVarReference.pmids_for_variant("")
    assert result == {}

    # No results for variant.
    mocker.patch("lib.evagg.ref.litvar.LitVarReference._requests_get_text", return_value=pmids_for_variant_fail)
    result = LitVarReference.pmids_for_variant("litvar%40rs113488022%23%23")


# def test_rate_limiting(mocker, pmids_for_variant_fail):
#     mocker.patch("lib.evagg.ref.litvar.LitVarReference._requests_get_text", return_value=pmids_for_variant_fail)

#     # Empty the call queue.
#     time.sleep(1)

#     # Record start time.
#     start = time.time()

#     # With no more than 3 calls per second, 7 calls should take at least 2 seconds.
#     for _ in range(7):
#         LitVarReference.pmids_for_variant("litvar%40rs113488022%23%23")

#     # Record end time.
#     end = time.time()

#     # Confirm that the calls took at least 2 seconds.
#     assert end - start >= 2
