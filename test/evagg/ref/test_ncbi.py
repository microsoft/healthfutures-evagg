from typing import Any, Optional

import pytest

from lib.evagg.ref import NcbiLookupClient
from lib.evagg.svc import IWebContentClient


@pytest.fixture
def web_client_result():
    class MockWebClient(IWebContentClient):
        def __init__(self, *responses) -> None:
            self._responses = iter(responses)

        def get(self, url: str, content_type: Optional[str] = None) -> Any:
            return next(self._responses)

    return MockWebClient


@pytest.fixture
def web_client_file_result(xml_load, json_load, web_client_result):
    def _web_client(*files):
        results = [xml_load(file) if file.endswith(".xml") else json_load(file) for file in files]
        return web_client_result(*results)

    return _web_client


def test_single_gene_direct(web_client_file_result):
    web_client = web_client_file_result("ncbi_symbol_single.json")
    result = NcbiLookupClient(web_client).gene_id_for_symbol("FAM111B")
    assert result == {"FAM111B": 374393}


def test_single_gene_indirect(web_client_file_result):
    web_client = web_client_file_result(*(["ncbi_symbol_synonym.json"] * 3))

    result = NcbiLookupClient(web_client).gene_id_for_symbol("FEB11")
    assert result == {}

    result = NcbiLookupClient(web_client).gene_id_for_symbol("FEB11", allow_synonyms=True)
    assert result == {"FEB11": 57094}

    # Verify that this would also work for the direct match.
    result = NcbiLookupClient(web_client).gene_id_for_symbol("CPA6")
    assert result == {"CPA6": 57094}


def test_single_gene_miss(web_client_result, web_client_file_result):
    web_client = web_client_result({})
    result = NcbiLookupClient(web_client).gene_id_for_symbol("FAM11B")
    assert result == {}

    web_client = web_client_file_result("ncbi_symbol_single.json")
    result = NcbiLookupClient(web_client).gene_id_for_symbol("not a gene")
    assert result == {}


def test_multi_gene(web_client_file_result):
    web_client = web_client_file_result(*(["ncbi_symbol_multi.json"] * 3))

    result = NcbiLookupClient(web_client).gene_id_for_symbol(["FAM111B", "FEB11"])
    assert result == {"FAM111B": 374393}

    result = NcbiLookupClient(web_client).gene_id_for_symbol(["FAM111B", "FEB11"], allow_synonyms=True)
    assert result == {"FAM111B": 374393, "FEB11": 57094}

    result = NcbiLookupClient(web_client).gene_id_for_symbol(["FAM111B", "FEB11", "not a gene"], allow_synonyms=True)
    assert result == {"FAM111B": 374393, "FEB11": 57094}


def test_variant(web_client_file_result):
    web_client = web_client_file_result(*(["efetch_snp_single_variant.xml"] * 3))

    result = NcbiLookupClient(web_client).hgvs_from_rsid(["rs146010120"])
    assert result == {"rs146010120": {"hgvs_c": "NM_001276.4:c.104G>A", "hgvs_p": "NP_001267.2:p.Arg35Gln"}}

    result = NcbiLookupClient(web_client).hgvs_from_rsid("rs146010120")
    assert result == {"rs146010120": {"hgvs_c": "NM_001276.4:c.104G>A", "hgvs_p": "NP_001267.2:p.Arg35Gln"}}


def test_multi_variant(web_client_file_result):
    web_client = web_client_file_result("efetch_snp_multi_variant.xml")

    result = NcbiLookupClient(web_client).hgvs_from_rsid(["rs146010120", "rs113488022"])
    assert result == {
        "rs113488022": {"hgvs_p": "NP_004324.2:p.Val600Gly", "hgvs_c": "NM_004333.6:c.1799T>G"},
        "rs146010120": {"hgvs_p": "NP_001267.2:p.Arg35Gln", "hgvs_c": "NM_001276.4:c.104G>A"},
    }


def test_missing_variant(web_client_result):
    web_client = web_client_result(None)

    result = NcbiLookupClient(web_client).hgvs_from_rsid("rs123456789")
    assert result == {"rs123456789": {}}


def test_non_rsid(web_client_result):
    web_client = web_client_result("")

    with pytest.raises(ValueError):
        NcbiLookupClient(web_client).hgvs_from_rsid(["not a rsid"])
    with pytest.raises(ValueError):
        NcbiLookupClient(web_client).hgvs_from_rsid(["rs1a2b"])
    with pytest.raises(ValueError):
        NcbiLookupClient(web_client).hgvs_from_rsid(["12345"])


def test_pubmed_search(web_client_file_result, json_load):
    web_client = web_client_file_result("esearch_pubmed_gene_CPA6.xml")
    result = NcbiLookupClient(web_client).search("CPA6")
    assert result == ["24290490"]


def test_pubmed_fetch(web_client_file_result, json_load):
    web_client = web_client_file_result("efetch_pubmed_paper_24290490.xml")
    result = NcbiLookupClient(web_client).fetch("24290490")
    assert result and result.props == json_load("efetch_paper_24290490.json")


def test_pubmed_pmc_oa_fetch(web_client_file_result, json_load):
    web_client = web_client_file_result("efetch_pubmed_paper_31427284.xml", "ncbi_pmc_is_oa_PMC6824399.xml")
    result = NcbiLookupClient(web_client).fetch("31427284")
    assert result and result.props["is_pmc_oa"] is False

    web_client = web_client_file_result("efetch_pubmed_paper_31427284.xml", "ncbi_pmc_is_oa_PMC3564958.xml")
    result = NcbiLookupClient(web_client).fetch("31427284")
    assert result and result.props["is_pmc_oa"] is True


def test_pubmed_fetch_missing(web_client_result, xml_parse):
    web_client = web_client_result(None)
    result = NcbiLookupClient(web_client).fetch("7777777777777777")
    assert result is None
    web_client = web_client_result(xml_parse("<PubmedArticleSet></PubmedArticleSet>"))
    result = NcbiLookupClient(web_client).fetch("7777777777777777")
    assert result is None
