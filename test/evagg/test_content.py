import json
from typing import Any, Dict, List, Tuple

import pytest

from lib.evagg import PromptBasedContentExtractor, SimpleContentExtractor
from lib.evagg.content import IFindObservations, Observation, TextSection
from lib.evagg.llm import IPromptClient
from lib.evagg.ref import IFetchHPO, ISearchHPO
from lib.evagg.types import HGVSVariant, Paper


@pytest.fixture
def mock_prompt(mock_client: type) -> IPromptClient:
    return mock_client(IPromptClient)


@pytest.fixture
def mock_observation(mock_client: type) -> IFindObservations:
    return mock_client(IFindObservations)


@pytest.fixture
def mock_phenotype_fetcher(mock_client: type) -> IFetchHPO:
    return mock_client(IFetchHPO)


@pytest.fixture
def mock_phenotype_searcher(mock_client: type) -> ISearchHPO:
    return mock_client(ISearchHPO)


@pytest.fixture
def paper() -> Paper:
    return Paper(
        id="12345678",
        citation="Doe, J. et al. Test Journal 2021",
        abstract="This is a test paper.",
        pmcid="PMC123",
        can_access=True,
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
    paper: Paper, mock_prompt: Any, mock_observation: Any, mock_phenotype_searcher: Any, mock_phenotype_fetcher: Any
) -> None:
    fields = {
        "gene": "CHI3L1",
        "paper_id": "12345678",
        "hgvs_c": "c.1234A>G",
        "hgvs_p": "NA",
        "individual_id": "unknown",
        "phenotype": "test (HP:0123)",
        "zygosity": "test",
        "variant_inheritance": "test",
    }

    observation = Observation(
        variant=HGVSVariant(fields["hgvs_c"], fields["gene"], "transcript", True, True, None),
        individual="unknown",
        texts=[TextSection("TEST", "test", 0, "Here is the observation text.")],
        variant_descriptions=["c.1234A>G"],
        patient_descriptions=["unknown"],
    )

    prompts = mock_prompt(
        json.dumps({"zygosity": fields["zygosity"]}),
        json.dumps({"variant_inheritance": fields["variant_inheritance"]}),
        json.dumps({"phenotypes": ["test"]}),  # phenotypes_all
        json.dumps({"phenotypes": ["test"]}),  # phenotypes_observation
        json.dumps({"phenotypes": ["test"]}),  # phenotypes_acronyms
        json.dumps({"matched": ["test (HP:0123)"], "unmatched": []}),
    )
    pheno_searcher = mock_phenotype_searcher([{"id": "HP:0123", "name": "test"}])
    pheno_fetcher = mock_phenotype_fetcher()
    content_extractor = PromptBasedContentExtractor(
        list(fields.keys()), prompts, mock_observation([observation]), pheno_searcher, pheno_fetcher
    )
    content = content_extractor.extract(paper, fields["gene"])

    assert prompts.call_count("prompt_file") == 5
    assert len(content) == 1
    print("CONTENT")
    print(content[0])
    print("FIELDS")
    print(fields)
    assert content[0] == fields


def test_prompt_based_content_extractor_failures(
    paper: Paper, mock_prompt: Any, mock_observation: Any, mock_phenotype_searcher: Any, mock_phenotype_fetcher: Any
) -> None:
    fields = ["not a field"]
    observation: Dict[Tuple[HGVSVariant, str], List[str]] = {}
    with pytest.raises(ValueError):
        PromptBasedContentExtractor(
            fields, mock_prompt(), mock_observation(observation), mock_phenotype_searcher(), mock_phenotype_fetcher()
        )
