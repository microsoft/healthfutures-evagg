import os
import tempfile
from typing import Any

import pytest

from lib.evagg.ref import IRefSeqLookupClient, NcbiReferenceLookupClient
from lib.evagg.utils import IWebContentClient


@pytest.fixture
def mock_web_client(mock_client: Any) -> Any:
    return mock_client(IWebContentClient)


@pytest.fixture
def client(mock_web_client: Any) -> IRefSeqLookupClient:
    return NcbiReferenceLookupClient(reference_dir="test/resources", web_client=mock_web_client())


def test_transcript_accession_found(client: IRefSeqLookupClient) -> None:
    assert client.transcript_accession_for_symbol("A1CF") == "NM_999999.4"


def test_transcript_accession_not_found(client: IRefSeqLookupClient) -> None:
    assert client.transcript_accession_for_symbol("NOT_A_GENE") is None


def test_transcript_accession_no_primary_seq(client: IRefSeqLookupClient) -> None:
    assert client.transcript_accession_for_symbol("A2M") is None


def test_transcript_accession_multiple_reference_standard(client: IRefSeqLookupClient) -> None:
    assert client.transcript_accession_for_symbol("AASDH") == "NM_999999.4"


def test_protein_accession_found(client: IRefSeqLookupClient) -> None:
    assert client.protein_accession_for_symbol("A1CF") == "NP_999999.2"


def test_protein_accession_not_found(client: IRefSeqLookupClient) -> None:
    assert client.protein_accession_for_symbol("NOT_A_GENE") is None


def test_protein_accession_no_primary_seq(client: IRefSeqLookupClient) -> None:
    assert client.protein_accession_for_symbol("A2M") is None


def test_protein_accession_multiple_reference_standard(client: IRefSeqLookupClient) -> None:
    assert client.protein_accession_for_symbol("AASDH") == "NP_999999.2"


def test_resource_caching(mock_web_client: Any) -> None:

    with open("test/resources/LRG_RefSeqGene.tsv", "r") as f:
        web_client = mock_web_client(f.read())

    with tempfile.TemporaryDirectory() as temp_dir:

        # First, remove the actual temp_dir so that we're forced to create it.
        os.rmdir(temp_dir)

        client = NcbiReferenceLookupClient(reference_dir=temp_dir, web_client=web_client)

        # Shouldn't exist yet.
        assert not os.path.exists(os.path.join(temp_dir, "LRG_RefSeqGene.tsv"))

        assert client.protein_accession_for_symbol("AASDH") == "NP_999999.2"

        # Should exist now.
        assert os.path.exists(os.path.join(temp_dir, "LRG_RefSeqGene.tsv"))
