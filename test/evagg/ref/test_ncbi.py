import xml.etree.ElementTree as Et
from typing import Any, Optional

import pytest

from lib.evagg.ref import NcbiLookupClient
from lib.evagg.svc import IWebContentClient


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


class MockWebClient(IWebContentClient):
    def __init__(self, response: Any) -> None:
        self._response = response

    def get(self, url: str, content_type: Optional[str] = None) -> Any:
        return self._response


def test_single_gene_direct(mocker, single_gene_direct_match):
    web_client = MockWebClient(single_gene_direct_match)
    result = NcbiLookupClient(web_client).gene_id_for_symbol("FAM111B")
    assert result == {"FAM111B": 374393}


def test_single_gene_indirect(mocker, single_gene_indirect_match):
    web_client = MockWebClient(single_gene_indirect_match)

    result = NcbiLookupClient(web_client).gene_id_for_symbol("FEB11")
    assert result == {}

    result = NcbiLookupClient(web_client).gene_id_for_symbol("FEB11", allow_synonyms=True)
    assert result == {"FEB11": 57094}

    # Verify that this would also work for the direct match.
    result = NcbiLookupClient(web_client).gene_id_for_symbol("CPA6")
    assert result == {"CPA6": 57094}


def test_single_gene_miss(mocker, single_gene_miss, single_gene_direct_match):
    web_client = MockWebClient(single_gene_miss)
    result = NcbiLookupClient(web_client).gene_id_for_symbol("FAM11B")
    assert result == {}

    web_client = MockWebClient(single_gene_direct_match)
    result = NcbiLookupClient(web_client).gene_id_for_symbol("not a gene")
    assert result == {}


def test_multi_gene(mocker, multi_gene):
    web_client = MockWebClient(multi_gene)

    result = NcbiLookupClient(web_client).gene_id_for_symbol(["FAM111B", "FEB11"])
    assert result == {"FAM111B": 374393}

    result = NcbiLookupClient(web_client).gene_id_for_symbol(["FAM111B", "FEB11"], allow_synonyms=True)
    assert result == {"FAM111B": 374393, "FEB11": 57094}

    result = NcbiLookupClient(web_client).gene_id_for_symbol(["FAM111B", "FEB11", "not a gene"], allow_synonyms=True)
    assert result == {"FAM111B": 374393, "FEB11": 57094}


@pytest.fixture
def valid_variant():
    return '<?xml version="1.0" ?>\n<ExchangeSet xmlns:xsi="https://www.w3.org/2001/XMLSchema-instance" xmlns="https://www.ncbi.nlm.nih.gov/SNP/docsum" xsi:schemaLocation="https://www.ncbi.nlm.nih.gov/SNP/docsum ftp://ftp.ncbi.nlm.nih.gov/snp/specs/docsum_eutils.xsd" ><DocumentSummary uid="146010120"><SNP_ID>146010120</SNP_ID><ALLELE_ORIGIN/><GLOBAL_MAFS><MAF><STUDY>1000Genomes</STUDY><FREQ>T=0.000156/1</FREQ></MAF><MAF><STUDY>ALSPAC</STUDY><FREQ>T=0.000778/3</FREQ></MAF><MAF><STUDY>Estonian</STUDY><FREQ>T=0.004911/22</FREQ></MAF><MAF><STUDY>ExAC</STUDY><FREQ>T=0.001796/218</FREQ></MAF><MAF><STUDY>FINRISK</STUDY><FREQ>T=0.003289/1</FREQ></MAF><MAF><STUDY>GENOME_DK</STUDY><FREQ>T=0.025/1</FREQ></MAF><MAF><STUDY>GnomAD</STUDY><FREQ>T=0.001227/172</FREQ></MAF><MAF><STUDY>GnomAD_exomes</STUDY><FREQ>T=0.001372/345</FREQ></MAF><MAF><STUDY>GoESP</STUDY><FREQ>T=0.001153/15</FREQ></MAF><MAF><STUDY>NorthernSweden</STUDY><FREQ>T=0.001667/1</FREQ></MAF><MAF><STUDY>PAGE_STUDY</STUDY><FREQ>T=0.000229/18</FREQ></MAF><MAF><STUDY>TOPMED</STUDY><FREQ>T=0.00062/164</FREQ></MAF><MAF><STUDY>TWINSUK</STUDY><FREQ>T=0.000539/2</FREQ></MAF><MAF><STUDY>ALFA</STUDY><FREQ>T=0.001385/78</FREQ></MAF></GLOBAL_MAFS><GLOBAL_POPULATION/><GLOBAL_SAMPLESIZE>0</GLOBAL_SAMPLESIZE><SUSPECTED/><CLINICAL_SIGNIFICANCE/><GENES><GENE_E><NAME>CHI3L1</NAME><GENE_ID>1116</GENE_ID></GENE_E></GENES><ACC>NC_000001.11</ACC><CHR>1</CHR><HANDLE>GNOMAD,EVA_UK10K_TWINSUK,EVA_UK10K_ALSPAC,HUMAN_LONGEVITY,EVA_EXAC,1000G_HIGH_COVERAGE,EVA_GENOME_DK,EGCUT_WGS,TOPMED,AFFY,1000GENOMES,NHLBI-ESP,SWEGEN,ACPOP,EVA,EVA_FINRISK,EVA_DECODE,ILLUMINA,CLINSEQ_SNP,HUGCELL_USP,PAGE_CC,EXOME_CHIP</HANDLE><SPDI>NC_000001.11:203185336:C:T</SPDI><FXN_CLASS>missense_variant,coding_sequence_variant</FXN_CLASS><VALIDATED>by-frequency,by-alfa,by-cluster</VALIDATED><DOCSUM>HGVS=NC_000001.11:g.203185337C&gt;T,NC_000001.10:g.203154465C&gt;T,NG_013056.1:g.6458G&gt;A,NM_001276.4:c.104G&gt;A,NM_001276.3:c.104G&gt;A,NM_001276.2:c.104G&gt;A,XM_047442873.1:c.104G&gt;A,XM_047442840.1:c.104G&gt;A,XM_047442841.1:c.104G&gt;A,XM_047442846.1:c.104G&gt;A,XM_047442847.1:c.104G&gt;A,XM_047442848.1:c.104G&gt;A,XM_047442879.1:c.104G&gt;A,NP_001267.2:p.Arg35Gln,XP_047298829.1:p.Arg35Gln,XP_047298796.1:p.Arg35Gln,XP_047298797.1:p.Arg35Gln,XP_047298802.1:p.Arg35Gln,XP_047298803.1:p.Arg35Gln,XP_047298804.1:p.Arg35Gln,XP_047298835.1:p.Arg35Gln|SEQ=[C/T]|LEN=1|GENE=CHI3L1:1116</DOCSUM><TAX_ID>9606</TAX_ID><ORIG_BUILD>134</ORIG_BUILD><UPD_BUILD>156</UPD_BUILD><CREATEDATE>2011/05/09 23:43</CREATEDATE><UPDATEDATE>2022/10/12 18:03</UPDATEDATE><SS>342022629,488696792,491308867,491613118,1294274417,1574614274,1584014242,1585438669,1601936249,1644930282,1686004025,1958348051,2169321082,2732197647,2746539239,2765159644,2984890957,2988295171,3021169973,3651512840,3653660589,3656254590,3688442981,3725096415,3727779247,3770864823,3823705026,4480289588,5324189141,5445937732,5519401552,5833130357,5847569853,5848282907,5911594779,5939091154,5979301185</SS><ALLELE>Y</ALLELE><SNP_CLASS>snv</SNP_CLASS><CHRPOS>1:203185337</CHRPOS><CHRPOS_PREV_ASSM>1:203154465</CHRPOS_PREV_ASSM><TEXT/><SNP_ID_SORT>0146010120</SNP_ID_SORT><CLINICAL_SORT>0</CLINICAL_SORT><CITED_SORT/><CHRPOS_SORT>0203185337</CHRPOS_SORT><MERGED_SORT>0</MERGED_SORT></DocumentSummary>\n</ExchangeSet>'  # noqa: E501


@pytest.fixture
def valid_multi_variant():
    return '<?xml version="1.0" ?>\n<ExchangeSet xmlns:xsi="https://www.w3.org/2001/XMLSchema-instance" xmlns="https://www.ncbi.nlm.nih.gov/SNP/docsum" xsi:schemaLocation="https://www.ncbi.nlm.nih.gov/SNP/docsum ftp://ftp.ncbi.nlm.nih.gov/snp/specs/docsum_eutils.xsd" ><DocumentSummary uid="146010120"><SNP_ID>146010120</SNP_ID><ALLELE_ORIGIN/><GLOBAL_MAFS><MAF><STUDY>1000Genomes</STUDY><FREQ>T=0.000156/1</FREQ></MAF><MAF><STUDY>ALSPAC</STUDY><FREQ>T=0.000778/3</FREQ></MAF><MAF><STUDY>Estonian</STUDY><FREQ>T=0.004911/22</FREQ></MAF><MAF><STUDY>ExAC</STUDY><FREQ>T=0.001796/218</FREQ></MAF><MAF><STUDY>FINRISK</STUDY><FREQ>T=0.003289/1</FREQ></MAF><MAF><STUDY>GENOME_DK</STUDY><FREQ>T=0.025/1</FREQ></MAF><MAF><STUDY>GnomAD</STUDY><FREQ>T=0.001227/172</FREQ></MAF><MAF><STUDY>GnomAD_exomes</STUDY><FREQ>T=0.001372/345</FREQ></MAF><MAF><STUDY>GoESP</STUDY><FREQ>T=0.001153/15</FREQ></MAF><MAF><STUDY>NorthernSweden</STUDY><FREQ>T=0.001667/1</FREQ></MAF><MAF><STUDY>PAGE_STUDY</STUDY><FREQ>T=0.000229/18</FREQ></MAF><MAF><STUDY>TOPMED</STUDY><FREQ>T=0.00062/164</FREQ></MAF><MAF><STUDY>TWINSUK</STUDY><FREQ>T=0.000539/2</FREQ></MAF><MAF><STUDY>ALFA</STUDY><FREQ>T=0.001385/78</FREQ></MAF></GLOBAL_MAFS><GLOBAL_POPULATION/><GLOBAL_SAMPLESIZE>0</GLOBAL_SAMPLESIZE><SUSPECTED/><CLINICAL_SIGNIFICANCE/><GENES><GENE_E><NAME>CHI3L1</NAME><GENE_ID>1116</GENE_ID></GENE_E></GENES><ACC>NC_000001.11</ACC><CHR>1</CHR><HANDLE>GNOMAD,EVA_UK10K_TWINSUK,EVA_UK10K_ALSPAC,HUMAN_LONGEVITY,EVA_EXAC,1000G_HIGH_COVERAGE,EVA_GENOME_DK,EGCUT_WGS,TOPMED,AFFY,1000GENOMES,NHLBI-ESP,SWEGEN,ACPOP,EVA,EVA_FINRISK,EVA_DECODE,ILLUMINA,CLINSEQ_SNP,HUGCELL_USP,PAGE_CC,EXOME_CHIP</HANDLE><SPDI>NC_000001.11:203185336:C:T</SPDI><FXN_CLASS>missense_variant,coding_sequence_variant</FXN_CLASS><VALIDATED>by-frequency,by-alfa,by-cluster</VALIDATED><DOCSUM>HGVS=NC_000001.11:g.203185337C&gt;T,NC_000001.10:g.203154465C&gt;T,NG_013056.1:g.6458G&gt;A,NM_001276.4:c.104G&gt;A,NM_001276.3:c.104G&gt;A,NM_001276.2:c.104G&gt;A,XM_047442873.1:c.104G&gt;A,XM_047442840.1:c.104G&gt;A,XM_047442841.1:c.104G&gt;A,XM_047442846.1:c.104G&gt;A,XM_047442847.1:c.104G&gt;A,XM_047442848.1:c.104G&gt;A,XM_047442879.1:c.104G&gt;A,NP_001267.2:p.Arg35Gln,XP_047298829.1:p.Arg35Gln,XP_047298796.1:p.Arg35Gln,XP_047298797.1:p.Arg35Gln,XP_047298802.1:p.Arg35Gln,XP_047298803.1:p.Arg35Gln,XP_047298804.1:p.Arg35Gln,XP_047298835.1:p.Arg35Gln|SEQ=[C/T]|LEN=1|GENE=CHI3L1:1116</DOCSUM><TAX_ID>9606</TAX_ID><ORIG_BUILD>134</ORIG_BUILD><UPD_BUILD>156</UPD_BUILD><CREATEDATE>2011/05/09 23:43</CREATEDATE><UPDATEDATE>2022/10/12 18:03</UPDATEDATE><SS>342022629,488696792,491308867,491613118,1294274417,1574614274,1584014242,1585438669,1601936249,1644930282,1686004025,1958348051,2169321082,2732197647,2746539239,2765159644,2984890957,2988295171,3021169973,3651512840,3653660589,3656254590,3688442981,3725096415,3727779247,3770864823,3823705026,4480289588,5324189141,5445937732,5519401552,5833130357,5847569853,5848282907,5911594779,5939091154,5979301185</SS><ALLELE>Y</ALLELE><SNP_CLASS>snv</SNP_CLASS><CHRPOS>1:203185337</CHRPOS><CHRPOS_PREV_ASSM>1:203154465</CHRPOS_PREV_ASSM><TEXT/><SNP_ID_SORT>0146010120</SNP_ID_SORT><CLINICAL_SORT>0</CLINICAL_SORT><CITED_SORT/><CHRPOS_SORT>0203185337</CHRPOS_SORT><MERGED_SORT>0</MERGED_SORT></DocumentSummary>\n<DocumentSummary uid="113488022"><SNP_ID>113488022</SNP_ID><ALLELE_ORIGIN/><GLOBAL_MAFS><MAF><STUDY>ExAC</STUDY><FREQ>T=0.000016/2</FREQ></MAF><MAF><STUDY>GnomAD_exomes</STUDY><FREQ>T=0.000004/1</FREQ></MAF></GLOBAL_MAFS><GLOBAL_POPULATION/><GLOBAL_SAMPLESIZE>0</GLOBAL_SAMPLESIZE><SUSPECTED/><CLINICAL_SIGNIFICANCE>pathogenic,other,likely-pathogenic,conflicting-interpretations-of-pathogenicity,drug-response</CLINICAL_SIGNIFICANCE><GENES><GENE_E><NAME>BRAF</NAME><GENE_ID>673</GENE_ID></GENE_E></GENES><ACC>NC_000007.14</ACC><CHR>7</CHR><HANDLE>EVA_EXAC,CPQ_GEN_INCA,OMIM-CURATED-RECORDS,YSAMUELS,EVA,MPIMG-CANCERGENOMICS,CLINVAR,DF-BWCC,GNOMAD,CSS-BFX</HANDLE><SPDI>NC_000007.14:140753335:A:C,NC_000007.14:140753335:A:G,NC_000007.14:140753335:A:T</SPDI><FXN_CLASS>intron_variant,missense_variant,coding_sequence_variant</FXN_CLASS><VALIDATED>by-frequency,by-cluster</VALIDATED><DOCSUM>HGVS=NC_000007.14:g.140753336A&gt;C,NC_000007.14:g.140753336A&gt;G,NC_000007.14:g.140753336A&gt;T,NC_000007.13:g.140453136A&gt;C,NC_000007.13:g.140453136A&gt;G,NC_000007.13:g.140453136A&gt;T,NG_007873.3:g.176429T&gt;G,NG_007873.3:g.176429T&gt;C,NG_007873.3:g.176429T&gt;A,NM_004333.6:c.1799T&gt;G,NM_004333.6:c.1799T&gt;C,NM_004333.6:c.1799T&gt;A,NM_004333.5:c.1799T&gt;G,NM_004333.5:c.1799T&gt;C,NM_004333.5:c.1799T&gt;A,NM_004333.4:c.1799T&gt;G,NM_004333.4:c.1799T&gt;C,NM_004333.4:c.1799T&gt;A,NM_001354609.2:c.1799T&gt;G,NM_001354609.2:c.1799T&gt;C,NM_001354609.2:c.1799T&gt;A,NM_001354609.1:c.1799T&gt;G,NM_001354609.1:c.1799T&gt;C,NM_001354609.1:c.1799T&gt;A,NM_001374258.1:c.1919T&gt;G,NM_001374258.1:c.1919T&gt;C,NM_001374258.1:c.1919T&gt;A,NM_001378467.1:c.1808T&gt;G,NM_001378467.1:c.1808T&gt;C,NM_001378467.1:c.1808T&gt;A,NM_001378470.1:c.1697T&gt;G,NM_001378470.1:c.1697T&gt;C,NM_001378470.1:c.1697T&gt;A,NM_001378471.1:c.1688T&gt;G,NM_001378471.1:c.1688T&gt;C,NM_001378471.1:c.1688T&gt;A,NM_001378468.1:c.1799T&gt;G,NM_001378468.1:c.1799T&gt;C,NM_001378468.1:c.1799T&gt;A,NM_001378475.1:c.1535T&gt;G,NM_001378475.1:c.1535T&gt;C,NM_001378475.1:c.1535T&gt;A,NM_001378472.1:c.1643T&gt;G,NM_001378472.1:c.1643T&gt;C,NM_001378472.1:c.1643T&gt;A,NM_001374244.1:c.1919T&gt;G,NM_001374244.1:c.1919T&gt;C,NM_001374244.1:c.1919T&gt;A,NM_001378469.1:c.1733T&gt;G,NM_001378469.1:c.1733T&gt;C,NM_001378469.1:c.1733T&gt;A,NM_001378473.1:c.1643T&gt;G,NM_001378473.1:c.1643T&gt;C,NM_001378473.1:c.1643T&gt;A,XM_017012559.2:c.1919T&gt;G,XM_017012559.2:c.1919T&gt;C,XM_017012559.2:c.1919T&gt;A,XM_017012559.1:c.1919T&gt;G,XM_017012559.1:c.1919T&gt;C,XM_017012559.1:c.1919T&gt;A,NR_148928.2:n.2898T&gt;G,NR_148928.2:n.2898T&gt;C,NR_148928.2:n.2898T&gt;A,XM_047420766.1:c.1763T&gt;G,XM_047420766.1:c.1763T&gt;C,XM_047420766.1:c.1763T&gt;A,XM_047420770.1:c.1085T&gt;G,XM_047420770.1:c.1085T&gt;C,XM_047420770.1:c.1085T&gt;A,NR_148928.1:n.2897T&gt;G,NR_148928.1:n.2897T&gt;C,NR_148928.1:n.2897T&gt;A,NM_001378474.1:c.1799T&gt;G,NM_001378474.1:c.1799T&gt;C,NM_001378474.1:c.1799T&gt;A,XM_047420767.1:c.1919T&gt;G,XM_047420767.1:c.1919T&gt;C,XM_047420767.1:c.1919T&gt;A,NP_004324.2:p.Val600Gly,NP_004324.2:p.Val600Ala,NP_004324.2:p.Val600Glu,NP_001341538.1:p.Val600Gly,NP_001341538.1:p.Val600Ala,NP_001341538.1:p.Val600Glu,NP_001361187.1:p.Val640Gly,NP_001361187.1:p.Val640Ala,NP_001361187.1:p.Val640Glu,NP_001365396.1:p.Val603Gly,NP_001365396.1:p.Val603Ala,NP_001365396.1:p.Val603Glu,NP_001365399.1:p.Val566Gly,NP_001365399.1:p.Val566Ala,NP_001365399.1:p.Val566Glu,NP_001365400.1:p.Val563Gly,NP_001365400.1:p.Val563Ala,NP_001365400.1:p.Val563Glu,NP_001365397.1:p.Val600Gly,NP_001365397.1:p.Val600Ala,NP_001365397.1:p.Val600Glu,NP_001365404.1:p.Val512Gly,NP_001365404.1:p.Val512Ala,NP_001365404.1:p.Val512Glu,NP_001365401.1:p.Val548Gly,NP_001365401.1:p.Val548Ala,NP_001365401.1:p.Val548Glu,NP_001361173.1:p.Val640Gly,NP_001361173.1:p.Val640Ala,NP_001361173.1:p.Val640Glu,NP_001365398.1:p.Val578Gly,NP_001365398.1:p.Val578Ala,NP_001365398.1:p.Val578Glu,NP_001365402.1:p.Val548Gly,NP_001365402.1:p.Val548Ala,NP_001365402.1:p.Val548Glu,XP_016868048.1:p.Val640Gly,XP_016868048.1:p.Val640Ala,XP_016868048.1:p.Val640Glu,XP_047276722.1:p.Val588Gly,XP_047276722.1:p.Val588Ala,XP_047276722.1:p.Val588Glu,XP_047276726.1:p.Val362Gly,XP_047276726.1:p.Val362Ala,XP_047276726.1:p.Val362Glu,NP_001365403.1:p.Val600Gly,NP_001365403.1:p.Val600Ala,NP_001365403.1:p.Val600Glu,XP_047276723.1:p.Val640Gly,XP_047276723.1:p.Val640Ala,XP_047276723.1:p.Val640Glu|SEQ=[A/C/G/T]|LEN=1|GENE=BRAF:673</DOCSUM><TAX_ID>9606</TAX_ID><ORIG_BUILD>132</ORIG_BUILD><UPD_BUILD>156</UPD_BUILD><CREATEDATE>2010/07/04 18:13</CREATEDATE><UPDATEDATE>2022/10/14 13:01</UPDATEDATE><SS>218178475,275515231,275515233,275518048,344939594,831879180,1688979043,1849950347,1849950348,1849950349,1849950350,1849950351,1849950352,1849950353,1849950354,1849950355,1849950356,1849950357,1849950358,1849950359,1849950360,1849950361,1849950362,1849950363,1849950364,1849950365,1849950366,1849950367,1849950368,1849950369,1849950370,1849950371,1849950372,1849950373,2736818121,5236855179,5442109030,5442109031,5442109032,5512434766,5935900606</SS><ALLELE>N</ALLELE><SNP_CLASS>snv</SNP_CLASS><CHRPOS>7:140753336</CHRPOS><CHRPOS_PREV_ASSM>7:140453136</CHRPOS_PREV_ASSM><TEXT/><SNP_ID_SORT>0113488022</SNP_ID_SORT><CLINICAL_SORT>1</CLINICAL_SORT><CITED_SORT/><CHRPOS_SORT>0140753336</CHRPOS_SORT><MERGED_SORT>0</MERGED_SORT></DocumentSummary>\n</ExchangeSet>'  # noqa: E501


def test_variant(valid_variant):
    web_client = MockWebClient(Et.fromstring(valid_variant))

    result = NcbiLookupClient(web_client).hgvs_from_rsid(["rs146010120"])
    assert result == {"rs146010120": {"hgvs_c": "NM_001276.4:c.104G>A", "hgvs_p": "NP_001267.2:p.Arg35Gln"}}

    result = NcbiLookupClient(web_client).hgvs_from_rsid("rs146010120")
    assert result == {"rs146010120": {"hgvs_c": "NM_001276.4:c.104G>A", "hgvs_p": "NP_001267.2:p.Arg35Gln"}}


def test_multi_variant(valid_multi_variant):
    web_client = MockWebClient(Et.fromstring(valid_multi_variant))

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
