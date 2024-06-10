import json
from typing import Any, Dict, List, Tuple

import pytest

from lib.evagg import PromptBasedContentExtractor
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
        variant=HGVSVariant(fields["hgvs_c"], fields["gene"], "transcript", True, True, None, []),
        individual="unknown",
        texts=[TextSection("TEST", "test", 0, "Here is the observation text.", "unknown")],
        variant_descriptions=["c.1234A>G"],
        patient_descriptions=["unknown"],
    )

    prompts = mock_prompt(
        json.dumps({"zygosity": fields["zygosity"]}),
        json.dumps({"variant_inheritance": fields["variant_inheritance"]}),
        json.dumps({"phenotypes": ["test"]}),  # phenotypes_all, only one text, so only once.
        json.dumps({"phenotypes": ["test"]}),  # phenotypes_observation, only one text, so only once.
        json.dumps({"phenotypes": ["test"]}),  # phenotypes_acronyms, only one text, so only once.
        json.dumps({"match": "test (HP:0123)"}),
    )
    pheno_searcher = mock_phenotype_searcher(
        [{"id": "HP:0123", "name": "test", "definition": "test", "synonyms": "test"}]
    )
    pheno_fetcher = mock_phenotype_fetcher()
    content_extractor = PromptBasedContentExtractor(
        list(fields.keys()), prompts, mock_observation([observation]), pheno_searcher, pheno_fetcher
    )
    content = content_extractor.extract(paper, fields["gene"])

    assert prompts.call_count("prompt_file") == 6
    assert len(content) == 1
    print("CONTENT")
    print(content[0])
    print("FIELDS")
    print(fields)
    assert content[0] == fields
