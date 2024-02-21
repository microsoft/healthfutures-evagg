import json

import pytest

from lib.evagg import PromptBasedContentExtractor, SimpleContentExtractor
from lib.evagg.content import IFindVariantMentions
from lib.evagg.llm import IPromptClient
from lib.evagg.ref import IVariantLookupClient
from lib.evagg.types import HGVSVariant, Paper


@pytest.fixture
def mock_prompt(mock_client: type) -> IPromptClient:
    return mock_client(IPromptClient)


@pytest.fixture
def mock_mention(mock_client: type) -> IFindVariantMentions:
    return mock_client(IFindVariantMentions)


@pytest.fixture
def mock_lookup(mock_client: type) -> IVariantLookupClient:
    return mock_client(IVariantLookupClient)


@pytest.fixture
def paper() -> Paper:
    return Paper(
        id="12345678", citation="Doe, J. et al. Test Journal 2021", abstract="This is a test paper.", pmcid="PMC123"
    )


def test_simple_content_extractor(paper) -> None:
    fields = ["gene", "hgvs_p", "variant_inheritance", "phenotype"]
    extractor = SimpleContentExtractor(fields)
    result = extractor.extract(paper, "CHI3L1")
    assert len(result) == 1
    assert result[0]["gene"] == "CHI3L1"
    assert result[0]["hgvs_p"] == "p.Y34C"
    assert result[0]["variant_inheritance"] == "AD"
    assert result[0]["phenotype"] == "Long face (HP:0000276)"


def test_prompt_based_content_extractor_valid_fields(paper, mock_prompt, mock_mention, mock_lookup) -> None:
    fields = {
        "gene": "CHI3L1",
        "paper_id": "12345678",
        "hgvs_c": "c.1234A>G",
        "hgvs_p": "c.1234A>G",
        "individual_id": "unknown",
        "phenotype": "test",
        "zygosity": "test",
        "variant_inheritance": "test",
    }
    mention = {
        HGVSVariant(fields["hgvs_c"], fields["gene"], "transcript", True, True): [
            {"text": paper.abstract, "gene_id": 1116, "gene_symbol": fields["gene"]}
        ]
    }
    lookup = {k.__str__(): {"hgvs_c": fields["hgvs_c"], "hgvs_p": fields["hgvs_p"]} for k in mention.keys()}
    prompts = mock_prompt(*[json.dumps({k: fields[k]}) for k in ["phenotype", "zygosity", "variant_inheritance"]])

    content_extractor = PromptBasedContentExtractor(
        list(fields.keys()), prompts, mock_mention(mention), mock_lookup(lookup)
    )
    content = content_extractor.extract(paper, fields["gene"])

    assert prompts.call_count("prompt_file") == 3
    assert len(content) == 1
    assert content[0] == fields


def test_prompt_based_content_extractor_failures(paper, mock_prompt, mock_mention, mock_lookup) -> None:
    fields = ["not a field"]
    mention = {"test": [{"text": paper.abstract, "gene_id": 1116, "gene_symbol": "test"}]}
    content_extractor = PromptBasedContentExtractor(fields, mock_prompt(), mock_mention(mention), mock_lookup())

    with pytest.raises(ValueError):
        content_extractor.extract(paper, "CHI3L1")

    paper.props.pop("pmcid")
    assert content_extractor.extract(paper, "CHI3L1") == []
