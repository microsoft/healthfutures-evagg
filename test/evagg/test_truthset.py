import os
from typing import Any, List

import pytest

from lib.di import DiContainer
from lib.evagg import TruthsetFileHandler
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import HGVSVariant, ICreateVariants, Paper


@pytest.fixture
def mock_variant(mock_client: type) -> ICreateVariants:
    return mock_client(ICreateVariants)


@pytest.fixture
def mock_paper_client(mock_client: type) -> IPaperLookupClient:
    return mock_client(IPaperLookupClient)


async def test_get_papers_error(mock_paper_client: Any, test_resources_path, json_load):
    path = os.path.join(test_resources_path, "truthset.tsv")
    paper = Paper(**json_load("paper_33057194.json"))
    paper_client = mock_paper_client(paper, None)

    library = TruthsetFileHandler(path, None, paper_client)  # type: ignore
    # On first call, mock paper client will return the wrong paper from fetch.
    with pytest.raises(ValueError) as e:
        library.get_papers({"gene_symbol": "BAZ2B"})
    assert "Truthset mismatch" in str(e.value)
    # On second call, mock paper client will return None.
    with pytest.raises(ValueError) as e:
        library.get_papers({"gene_symbol": "BAZ2B"})
    assert "Failed to fetch paper" in str(e.value)

    paper.props["can_access"] = False
    obs1 = await library.find_observations("BAZ2B", paper)
    assert not obs1


async def test_get_observations(mock_paper_client: Any, mock_variant: Any, test_resources_path, json_load) -> None:
    path = os.path.join(test_resources_path, "truthset.tsv")
    paper1 = Paper(**json_load("paper_28554332.json"))
    paper2 = Paper(**json_load("paper_33057194.json"))
    paper_client = mock_paper_client(paper1, paper2)

    variant0 = HGVSVariant("p.Arg81Gly", "BAZ2B", "NP_001276904.1", False, True, None, None, [])
    variant1 = HGVSVariant("c.242C>G", "BAZ2B", "NM_001289975.1", False, True, None, variant0, [])
    variant_client = mock_variant(variant1)

    library = TruthsetFileHandler(path, variant_client, paper_client)
    papers = library.get_papers({"gene_symbol": "BAZ2B"})
    assert {p.id for p in papers} == {"pmid:33057194", "pmid:28554332"}
    assert library.get_papers({}) == []

    obs1 = await library.find_observations("BAZ2B", papers[0])
    assert len(obs1) == 1
    assert obs1[0].variant == variant1
    assert obs1[0].patient_descriptions == ["00136-C"]
    assert len(obs1[0].texts) == 145


def test_get_evidence(mock_paper_client: Any, mock_variant: Any, test_resources_path):
    path = os.path.join(test_resources_path, "truthset.tsv")
    paper = Paper(id="pmid:33531950", citation="Test Citation", link="Test Link")
    paper_client = mock_paper_client(paper, paper)
    variant = HGVSVariant("c.1192C>T", "ZNF423", None, False, True, None, None, [])
    variant_client = mock_variant(variant, variant)

    fields: List = DiContainer().create_instance({"di_factory": "fields/evidence_all.yaml"}, {})
    library = TruthsetFileHandler(path, variant_client, paper_client, fields)
    fieldsets = library.extract(paper, "ZNF423")
    assert len(fieldsets) == 1
    assert fieldsets[0]["hgvs_c"] == "c.1192C>T"
    assert fieldsets[0]["gene"] == "ZNF423"
    assert fieldsets[0]["evidence_id"] == "pmid-33531950_c.1192C-T_patient"
    assert fieldsets[0]["variant_type"] == "missense"

    fields.append("missing_field")
    library = TruthsetFileHandler(path, variant_client, paper_client, fields)
    with pytest.raises(ValueError) as e:
        library.extract(paper, "ZNF423")
    assert "Unsupported extraction fields" in str(e.value)
