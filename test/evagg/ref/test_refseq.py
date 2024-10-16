import os
import tempfile
from typing import Any
from unittest.mock import patch

import pytest

from lib.evagg.ref import IRefSeqLookupClient, RefSeqGeneLookupClient, RefSeqLookupClient
from lib.evagg.utils import IWebContentClient


@pytest.fixture
def mock_web_client(mock_client: Any) -> Any:
    return mock_client(IWebContentClient)


@pytest.fixture
def rsg_client(mock_web_client: Any) -> IRefSeqLookupClient:
    return RefSeqGeneLookupClient(reference_dir="test/resources", web_client=mock_web_client())


@pytest.fixture
def rs_client(mock_web_client: Any) -> IRefSeqLookupClient:
    return RefSeqLookupClient(reference_dir="test/resources", web_client=mock_web_client())


def test_transcript_accession_found(rsg_client: IRefSeqLookupClient) -> None:
    assert rsg_client.transcript_accession_for_symbol("A1CF") == "NM_999999.4"


def test_transcript_accession_not_found(rsg_client: IRefSeqLookupClient) -> None:
    assert rsg_client.transcript_accession_for_symbol("NOT_A_GENE") is None


def test_transcript_accession_no_primary_seq(rsg_client: IRefSeqLookupClient) -> None:
    assert rsg_client.transcript_accession_for_symbol("A2M") is None


def test_transcript_accession_multiple_reference_standard(rsg_client: IRefSeqLookupClient) -> None:
    assert rsg_client.transcript_accession_for_symbol("AASDH") == "NM_999999.4"


def test_protein_accession_found(rsg_client: IRefSeqLookupClient) -> None:
    assert rsg_client.protein_accession_for_symbol("A1CF") == "NP_999999.2"


def test_protein_accession_not_found(rsg_client: IRefSeqLookupClient) -> None:
    assert rsg_client.protein_accession_for_symbol("NOT_A_GENE") is None


def test_protein_accession_no_primary_seq(rsg_client: IRefSeqLookupClient) -> None:
    assert rsg_client.protein_accession_for_symbol("A2M") is None


def test_protein_accession_multiple_reference_standard(rsg_client: IRefSeqLookupClient) -> None:
    assert rsg_client.protein_accession_for_symbol("AASDH") == "NP_999999.2"


def test_genomic_accession_found(rsg_client: IRefSeqLookupClient) -> None:
    assert rsg_client.genomic_accession_for_symbol("A1CF") == "NG_029916.1"


def test_genomic_accession_not_found(rsg_client: IRefSeqLookupClient) -> None:
    assert rsg_client.genomic_accession_for_symbol("NOT_A_GENE") is None


def test_accession_autocomplete_noop(rsg_client: IRefSeqLookupClient) -> None:
    assert rsg_client.accession_autocomplete("NM_999999.4") == "NM_999999.4"


def test_accession_autocomplete_found(mock_web_client: Any) -> None:
    web_client = mock_web_client("NM_138932.3\n")
    client = RefSeqGeneLookupClient(reference_dir="test/resources", web_client=web_client)
    assert client.accession_autocomplete("NM_138932") == "NM_138932.3"


def test_accession_autocomplete_not_found(mock_web_client: Any) -> None:
    web_client = mock_web_client(
        "Error: I D  l i s t  d o e s  n o t  c o n t a i n  v a l i d  I D s  o r  a c c e s s i o n s !\n"
    )
    client = RefSeqGeneLookupClient(reference_dir="test/resources", web_client=web_client)
    assert client.accession_autocomplete("NM_000000") is None


def test_refseq_use_cache(rs_client: IRefSeqLookupClient) -> None:
    assert rs_client.transcript_accession_for_symbol("OR4F29") == "NM_001005221.2"


def test_refseq_resource_caching(mock_web_client: Any) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock the _download_binary_reference method
        with patch.object(RefSeqLookupClient, "_download_binary_reference") as mock_download:

            def mock_copy_file(url: str, target: str) -> None:
                # This resource contains artificial duplicated rows to trigger some of the warning conditions.
                source = "test/resources/GCF_subset.gff.gz"
                with open(source, "rb") as src, open(target, "wb") as dst:
                    dst.write(src.read())

            mock_download.side_effect = mock_copy_file

            client = RefSeqLookupClient(reference_dir=temp_dir, web_client=mock_web_client())

            # Shouldn't exist yet.
            assert not os.path.exists(os.path.join(temp_dir, client._RAW_FILENAME))
            assert not os.path.exists(os.path.join(temp_dir, client._PROCESSED_FILEPATH))

            assert client.protein_accession_for_symbol("OR4F5") == "NP_001005484.2"
            assert client.transcript_accession_for_symbol("OR4F29") == "NM_001005221.2"

            # Should exist now.
            assert os.path.exists(os.path.join(temp_dir, client._PROCESSED_FILEPATH))

            # Shouldn't exist anymore.
            assert not os.path.exists(os.path.join(temp_dir, client._RAW_FILENAME))


def test_refseqgene_resource_caching(mock_web_client: Any) -> None:

    with open("test/resources/LRG_RefSeqGene.tsv", "r") as f:
        web_client = mock_web_client(f.read())

    with tempfile.TemporaryDirectory() as temp_dir:

        # First, remove the actual temp_dir so that we're forced to create it.
        os.rmdir(temp_dir)

        client = RefSeqGeneLookupClient(reference_dir=temp_dir, web_client=web_client)

        # Shouldn't exist yet.
        assert not os.path.exists(os.path.join(temp_dir, "LRG_RefSeqGene.tsv"))

        assert client.protein_accession_for_symbol("AASDH") == "NP_999999.2"

        # Should exist now.
        assert os.path.exists(os.path.join(temp_dir, "LRG_RefSeqGene.tsv"))
