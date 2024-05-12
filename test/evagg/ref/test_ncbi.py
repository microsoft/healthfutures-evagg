import pytest

from lib.evagg.content.fulltext import get_fulltext
from lib.evagg.ref import NcbiLookupClient
from lib.evagg.svc import IWebContentClient
from lib.evagg.types import Paper


@pytest.fixture
def mock_web_client(mock_client):
    return mock_client(IWebContentClient)


def test_single_gene_direct(mock_web_client):
    web_client = mock_web_client("ncbi_symbol_single.json")
    result = NcbiLookupClient(web_client).gene_id_for_symbol("FAM111B")
    assert result == {"FAM111B": 374393}


def test_single_gene_indirect(mock_web_client):
    web_client = mock_web_client(*(["ncbi_symbol_synonym.json"] * 3))

    result = NcbiLookupClient(web_client).gene_id_for_symbol("FEB11")
    assert result == {}

    result = NcbiLookupClient(web_client).gene_id_for_symbol("FEB11", allow_synonyms=True)
    assert result == {"FEB11": 57094}

    # Verify that this would also work for the direct match.
    result = NcbiLookupClient(web_client).gene_id_for_symbol("CPA6")
    assert result == {"CPA6": 57094}


def test_single_gene_miss(mock_web_client):
    web_client = mock_web_client({})
    result = NcbiLookupClient(web_client).gene_id_for_symbol("FAM11B")
    assert result == {}

    web_client = mock_web_client("ncbi_symbol_single.json")
    result = NcbiLookupClient(web_client).gene_id_for_symbol("not a gene")
    assert result == {}


def test_multi_gene(mock_web_client):
    web_client = mock_web_client(*(["ncbi_symbol_multi.json"] * 3))

    result = NcbiLookupClient(web_client).gene_id_for_symbol("FAM111B", "FEB11")
    assert result == {"FAM111B": 374393}

    result = NcbiLookupClient(web_client).gene_id_for_symbol("FAM111B", "FEB11", allow_synonyms=True)
    assert result == {"FAM111B": 374393, "FEB11": 57094}

    result = NcbiLookupClient(web_client).gene_id_for_symbol("FAM111B", "FEB11", "not a gene", allow_synonyms=True)
    assert result == {"FAM111B": 374393, "FEB11": 57094}


def test_variant(mock_web_client):
    web_client = mock_web_client(*(["efetch_snp_single_variant.xml"] * 3))

    result = NcbiLookupClient(web_client).hgvs_from_rsid("rs146010120")
    assert result == {"rs146010120": {"hgvs_c": "NM_001276.4:c.104G>A", "hgvs_p": "NP_001267.2:p.Arg35Gln"}}


def test_multi_variant(mock_web_client):
    web_client = mock_web_client("efetch_snp_multi_variant.xml")

    result = NcbiLookupClient(web_client).hgvs_from_rsid("rs146010120", "rs113488022")
    assert result == {
        "rs113488022": {"hgvs_p": "NP_004324.2:p.Val600Gly", "hgvs_c": "NM_004333.6:c.1799T>G"},
        "rs146010120": {"hgvs_p": "NP_001267.2:p.Arg35Gln", "hgvs_c": "NM_001276.4:c.104G>A"},
    }


def test_missing_variant(mock_web_client):
    web_client = mock_web_client(None)

    result = NcbiLookupClient(web_client).hgvs_from_rsid("rs123456789")
    assert result == {"rs123456789": {}}


def test_non_rsid(mock_web_client):
    web_client = mock_web_client("")

    with pytest.raises(ValueError):
        NcbiLookupClient(web_client).hgvs_from_rsid("not a rsid")
    with pytest.raises(ValueError):
        NcbiLookupClient(web_client).hgvs_from_rsid("rs1a2b")
    with pytest.raises(ValueError):
        NcbiLookupClient(web_client).hgvs_from_rsid("12345")


def test_pubmed_search(mock_web_client):
    web_client = mock_web_client("esearch_pubmed_gene_CPA6.xml")
    result = NcbiLookupClient(web_client).search("CPA6")
    assert result == ["24290490"]


def test_pubmed_fetch(mock_web_client, json_load):
    web_client = mock_web_client("efetch_pubmed_paper_24290490.xml")
    result = NcbiLookupClient(web_client).fetch("24290490")
    assert result and result.props == json_load("efetch_paper_24290490.json")


def test_pubmed_pmc_oa_fetch(mock_web_client):
    web_client = mock_web_client("efetch_pubmed_paper_31427284.xml", "ncbi_pmc_is_oa_PMC6824399.xml")
    result = NcbiLookupClient(web_client).fetch("31427284")
    assert result and result.props["can_access"] is False


def test_pubmed_pmc_full_text(mock_web_client):
    web_client = mock_web_client(
        "efetch_pubmed_paper_33688625.xml", "ncbi_pmc_is_oa_PMC7933980.xml", "ncbi_bioc_full_text_PMC7933980.xml"
    )
    result = NcbiLookupClient(web_client).fetch("33688625", include_fulltext=True)
    assert result and result.props["can_access"] is True
    assert (
        get_fulltext(result.props["fulltext_xml"], include=["TITLE"])
        == "Saul-Wilson Syndrome Missense Allele Does Not Show Obvious Golgi Defects in a C. elegans Model"
    )


def test_pubmed_fetch_missing(mock_web_client, xml_parse):
    web_client = mock_web_client(None)
    result = NcbiLookupClient(web_client).fetch("7777777777777777")
    assert result is None
    web_client = mock_web_client(xml_parse("<PubmedArticleSet></PubmedArticleSet>"))
    result = NcbiLookupClient(web_client).fetch("7777777777777777")
    assert result is None


def test_annotation(mock_web_client):
    web_client = mock_web_client({"foo": "bar"})
    paper = Paper(id="123", pmcid="PMC1234567", can_access=True)
    annotations = NcbiLookupClient(web_client).annotate(paper)
    assert isinstance(annotations, dict)
    assert annotations.get("foo") == "bar"
    assert web_client.call_count() == 1
    assert web_client.last_call("get") == (
        NcbiLookupClient.PUBTATOR_GET_URL.format(fmt="json", id="PMC1234567"),
        {"content_type": "json"},
    )


def test_no_annotation(mock_web_client):
    web_client = mock_web_client()

    paper = Paper(id="123", pmcid="PMC7654321")
    annotations = NcbiLookupClient(web_client).annotate(paper)
    assert annotations == {}

    paper_no_pmcid = Paper(id="123")
    annotations = NcbiLookupClient(web_client).annotate(paper_no_pmcid)
    assert annotations == {}

    paper_no_oa = Paper(id="123", pmcid="PMC1234567", can_access=False)
    annotations = NcbiLookupClient(web_client).annotate(paper_no_oa)
    assert annotations == {}

    assert web_client.call_count() == 0
