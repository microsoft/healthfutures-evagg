from typing import Any, Optional

import pytest

from lib.evagg.ref import NcbiLookupClient
from lib.evagg.svc import IWebContentClient


@pytest.fixture
def single_gene_miss():
    return {}


@pytest.fixture
def single_gene_direct_match(json_load):
    return json_load("ncbi_symbol_single.json")


@pytest.fixture
def single_gene_indirect_match(json_load):
    return json_load("ncbi_symbol_synonym.json")


@pytest.fixture
def multi_gene(json_load):
    return json_load("ncbi_symbol_multi.json")


@pytest.fixture
def single_variant(xml_load):
    return xml_load("efetch_snp_single_variant.xml")


@pytest.fixture
def multi_variant(xml_load):
    return xml_load("efetch_snp_multi_variant.xml")


class MockWebClient(IWebContentClient):
    def __init__(self, response: Any) -> None:
        self._response = response

    def get(self, url: str, content_type: Optional[str] = None) -> Any:
        return self._response


def test_single_gene_direct(single_gene_direct_match):
    web_client = MockWebClient(single_gene_direct_match)
    result = NcbiLookupClient(web_client).gene_id_for_symbol("FAM111B")
    assert result == {"FAM111B": 374393}


def test_single_gene_indirect(single_gene_indirect_match):
    web_client = MockWebClient(single_gene_indirect_match)

    result = NcbiLookupClient(web_client).gene_id_for_symbol("FEB11")
    assert result == {}

    result = NcbiLookupClient(web_client).gene_id_for_symbol("FEB11", allow_synonyms=True)
    assert result == {"FEB11": 57094}

    # Verify that this would also work for the direct match.
    result = NcbiLookupClient(web_client).gene_id_for_symbol("CPA6")
    assert result == {"CPA6": 57094}


def test_single_gene_miss(single_gene_miss, single_gene_direct_match):
    web_client = MockWebClient(single_gene_miss)
    result = NcbiLookupClient(web_client).gene_id_for_symbol("FAM11B")
    assert result == {}

    web_client = MockWebClient(single_gene_direct_match)
    result = NcbiLookupClient(web_client).gene_id_for_symbol("not a gene")
    assert result == {}


def test_multi_gene(multi_gene):
    web_client = MockWebClient(multi_gene)

    result = NcbiLookupClient(web_client).gene_id_for_symbol(["FAM111B", "FEB11"])
    assert result == {"FAM111B": 374393}

    result = NcbiLookupClient(web_client).gene_id_for_symbol(["FAM111B", "FEB11"], allow_synonyms=True)
    assert result == {"FAM111B": 374393, "FEB11": 57094}

    result = NcbiLookupClient(web_client).gene_id_for_symbol(["FAM111B", "FEB11", "not a gene"], allow_synonyms=True)
    assert result == {"FAM111B": 374393, "FEB11": 57094}


def test_variant(single_variant):
    web_client = MockWebClient(single_variant)

    result = NcbiLookupClient(web_client).hgvs_from_rsid(["rs146010120"])
    assert result == {"rs146010120": {"hgvs_c": "NM_001276.4:c.104G>A", "hgvs_p": "NP_001267.2:p.Arg35Gln"}}

    result = NcbiLookupClient(web_client).hgvs_from_rsid("rs146010120")
    assert result == {"rs146010120": {"hgvs_c": "NM_001276.4:c.104G>A", "hgvs_p": "NP_001267.2:p.Arg35Gln"}}


def test_multi_variant(multi_variant):
    web_client = MockWebClient(multi_variant)

    result = NcbiLookupClient(web_client).hgvs_from_rsid(["rs146010120", "rs113488022"])
    assert result == {
        "rs113488022": {"hgvs_p": "NP_004324.2:p.Val600Gly", "hgvs_c": "NM_004333.6:c.1799T>G"},
        "rs146010120": {"hgvs_p": "NP_001267.2:p.Arg35Gln", "hgvs_c": "NM_001276.4:c.104G>A"},
    }


def test_missing_variant():
    web_client = MockWebClient(None)

    result = NcbiLookupClient(web_client).hgvs_from_rsid("rs123456789")
    assert result == {}


def test_non_rsid():
    web_client = MockWebClient("")

    with pytest.raises(ValueError):
        NcbiLookupClient(web_client).hgvs_from_rsid(["not a rsid"])

    with pytest.raises(ValueError):
        NcbiLookupClient(web_client).hgvs_from_rsid(["rs1a2b"])

    with pytest.raises(ValueError):
        NcbiLookupClient(web_client).hgvs_from_rsid(["12345"])
