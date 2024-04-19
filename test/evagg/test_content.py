import json
from typing import Any, Dict, List, Tuple

import pytest

from lib.evagg import PromptBasedContentExtractor, SimpleContentExtractor
from lib.evagg.content import IFindObservations
from lib.evagg.llm import IPromptClient
from lib.evagg.ref import ISearchHPO
from lib.evagg.types import HGVSVariant, Paper


@pytest.fixture
def mock_prompt(mock_client: type) -> IPromptClient:
    return mock_client(IPromptClient)


@pytest.fixture
def mock_observation(mock_client: type) -> IFindObservations:
    return mock_client(IFindObservations)


@pytest.fixture
def mock_phenotype(mock_client: type) -> ISearchHPO:
    return mock_client(ISearchHPO)


@pytest.fixture
def paper() -> Paper:
    return Paper(
        id="12345678",
        citation="Doe, J. et al. Test Journal 2021",
        abstract="This is a test paper.",
        pmcid="PMC123",
        is_pmc_oa=True,
    )


def test_simple_content_extractor(paper: Paper) -> None:
    fields = ["gene", "hgvs_p", "variant_inheritance", "phenotype"]
    extractor = SimpleContentExtractor(fields)
    result = extractor.extract(paper, "CHI3L1")
    assert len(result) == 1
    assert result[0]["gene"] == "CHI3L1"
    assert result[0]["hgvs_p"] == "p.Y34C"
    assert result[0]["variant_inheritance"] == "AD"
    assert result[0]["phenotype"] == "Long face (HP:0000276)"


def test_prompt_based_content_extractor_valid_fields(
    paper: Paper, mock_prompt: Any, mock_observation: Any, mock_phenotype: Any
) -> None:
    fields = {
        "gene": "CHI3L1",
        "paper_id": "12345678",
        "hgvs_c": "c.1234A>G",
        "hgvs_p": "NA",
        "individual_id": "unknown",
        "phenotype": '["test (HP:0123)"]',
        "zygosity": "test",
        "variant_inheritance": "test",
    }

    observation = {
        (HGVSVariant(fields["hgvs_c"], fields["gene"], "transcript", True, True, None), "unknown"): {
            "texts": ["Here is the observation text."],
            "variant_descriptions": ["c.1234A>G"],
            "patient_descriptions": ["unknown"],
        }
    }
    prompts = mock_prompt(
        json.dumps({"phenotype": ["test"]}),
        json.dumps({"matched": ["test (HP:0123)"], "unmatched": []}),
        json.dumps({"zygosity": fields["zygosity"]}),
        json.dumps({"variant_inheritance": fields["variant_inheritance"]}),
    )
    pheno = mock_phenotype(True)
    content_extractor = PromptBasedContentExtractor(list(fields.keys()), prompts, mock_observation(observation), pheno)
    content = content_extractor.extract(paper, fields["gene"])

    assert prompts.call_count("prompt_file") == 4
    assert len(content) == 1
    assert content[0] == fields


def test_prompt_based_content_extractor_failures(
    paper: Paper, mock_prompt: Any, mock_observation: Any, mock_phenotype: Any
) -> None:
    fields = ["not a field"]
    observation: Dict[Tuple[HGVSVariant, str], List[str]] = {}
    with pytest.raises(ValueError):
        PromptBasedContentExtractor(fields, mock_prompt(), mock_observation(observation), mock_phenotype())
