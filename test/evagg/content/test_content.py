import json
from typing import Any

import pytest

from lib.evagg import PromptBasedContentExtractor
from lib.evagg.content import IFindObservations, Observation, TextSection
from lib.evagg.content.fulltext import get_fulltext
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
        title="Test Paper Title",
        abstract="This the abstract from a test paper.",
        pmcid="PMC123",
        can_access=True,
    )


def test_prompt_based_content_extractor_valid_fields(
    paper: Paper, mock_prompt: Any, mock_observation: Any, mock_phenotype_searcher: Any, mock_phenotype_fetcher: Any
) -> None:
    fields = {
        "evidence_id": "12345678_c.1234A-G_unknown",  # TODO
        "gene": "CHI3L1",
        "paper_id": "12345678",
        "citation": paper.props["citation"],
        "link": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{paper.props['pmcid']}",
        "paper_title": paper.props["title"],
        "hgvs_c": "c.1234A>G",
        "hgvs_p": "NA",
        "paper_variant": "c.1234A>G",
        "transcript": "transcript",
        "valid": "True",
        "validation_error": "not an error",
        "individual_id": "unknown",
        "gnomad_frequency": "unknown",
        "phenotype": "test (HP:0123)",
        "zygosity": "test",
        "variant_inheritance": "test",
    }

    observation = Observation(
        variant=HGVSVariant(
            hgvs_desc=fields["hgvs_c"],
            gene_symbol=fields["gene"],
            refseq=fields["transcript"],
            refseq_predicted=True,
            valid=fields["valid"] == "True",
            validation_error=fields["validation_error"],
            protein_consequence=None,
            coding_equivalents=[],
        ),
        individual="unknown",
        texts=[TextSection("TEST", "test", 0, "Here is the observation text.", "unknown")],
        variant_descriptions=fields["paper_variant"].split(", "),
        patient_descriptions=["unknown"],
        paper_id=fields["paper_id"],
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
    for key in fields:
        assert content[0][key] == fields[key]


def test_prompt_based_content_extractor_unsupported_field(
    paper: Paper, mock_prompt: Any, mock_observation: Any, mock_phenotype_searcher: Any, mock_phenotype_fetcher: Any
) -> None:
    fields = {"unsupported_field": "unsupported_value"}

    observation = Observation(
        variant=HGVSVariant(
            hgvs_desc="hgvs_desc",
            gene_symbol="gene",
            refseq="transcript",
            refseq_predicted=True,
            valid=True,
            validation_error="",
            protein_consequence=None,
            coding_equivalents=[],
        ),
        individual="unknown",
        texts=[TextSection("TEST", "test", 0, "Here is the observation text.", "unknown")],
        variant_descriptions=["hgvs_desc"],
        patient_descriptions=["unknown"],
        paper_id="paper_id",
    )

    content_extractor = PromptBasedContentExtractor(
        list(fields.keys()),
        mock_prompt({}),
        mock_observation([observation]),
        mock_phenotype_searcher([]),
        mock_phenotype_fetcher(),
    )
    with pytest.raises(ValueError):
        _ = content_extractor.extract(paper, "test")


def test_prompt_based_content_extractor_with_protein_consequence(
    paper: Paper, mock_prompt: Any, mock_observation: Any, mock_phenotype_searcher: Any, mock_phenotype_fetcher: Any
) -> None:
    fields = {
        "gene": "CHI3L1",
        "paper_id": "12345678",
        "hgvs_c": "c.1234A>G",
        "hgvs_p": "p.Ala123Gly",
        "paper_variant": "c.1234A>G, c.1234 A > G",
        "individual_id": "unknown",
    }

    protein_variant = HGVSVariant(fields["hgvs_p"], fields["gene"], "transcript", True, True, None, None, [])

    observation = Observation(
        variant=HGVSVariant(fields["hgvs_c"], fields["gene"], "transcript", True, True, None, protein_variant, []),
        individual=fields["individual_id"],
        texts=[TextSection("TEST", "test", 0, "Here is the observation text.", "unknown")],
        variant_descriptions=fields["paper_variant"].split(", "),
        patient_descriptions=[fields["individual_id"]],
        paper_id=fields["paper_id"],
    )

    prompts = mock_prompt({})
    pheno_searcher = mock_phenotype_searcher([])
    pheno_fetcher = mock_phenotype_fetcher()
    content_extractor = PromptBasedContentExtractor(
        list(fields.keys()), prompts, mock_observation([observation]), pheno_searcher, pheno_fetcher
    )
    content = content_extractor.extract(paper, fields["gene"])

    assert len(content) == 1
    for key in fields:
        assert content[0][key] == fields[key]


def test_prompt_based_content_extractor_invalid_model_response(
    paper: Paper, mock_prompt: Any, mock_observation: Any, mock_phenotype_searcher: Any, mock_phenotype_fetcher: Any
) -> None:
    fields = {
        "gene": "CHI3L1",
        "paper_id": "12345678",
        "hgvs_c": "c.1234A>G",
        "paper_variant": "c.1234A>G",
        "individual_id": "unknown",
        "zygosity": "failed",
    }

    observation = Observation(
        variant=HGVSVariant(fields["hgvs_c"], fields["gene"], "transcript", True, True, None, None, []),
        individual=fields["individual_id"],
        texts=[TextSection("TEST", "test", 0, "Here is the observation text.", "unknown")],
        variant_descriptions=fields["paper_variant"].split(", "),
        patient_descriptions=[fields["individual_id"]],
        paper_id=fields["paper_id"],
    )

    prompts = mock_prompt("{invalid json")
    pheno_searcher = mock_phenotype_searcher([])
    pheno_fetcher = mock_phenotype_fetcher()
    content_extractor = PromptBasedContentExtractor(
        list(fields.keys()), prompts, mock_observation([observation]), pheno_searcher, pheno_fetcher
    )
    content = content_extractor.extract(paper, fields["gene"])

    assert len(content) == 1
    for key in fields:
        assert content[0][key] == fields[key]


def test_prompt_based_content_extractor_phenotype_empty_list(
    paper: Paper, mock_prompt: Any, mock_observation: Any, mock_phenotype_searcher: Any, mock_phenotype_fetcher: Any
) -> None:
    fields = {
        "gene": "CHI3L1",
        "paper_id": "12345678",
        "hgvs_c": "c.1234A>G",
        "paper_variant": "c.1234A>G",
        "individual_id": "unknown",
        "phenotype": "",
    }

    observation = Observation(
        variant=HGVSVariant(fields["hgvs_c"], fields["gene"], "transcript", True, True, None, None, []),
        individual=fields["individual_id"],
        texts=[TextSection("TEST", "test", 0, "Here is the observation text.", "unknown")],
        variant_descriptions=fields["paper_variant"].split(", "),
        patient_descriptions=[fields["individual_id"]],
        paper_id=fields["paper_id"],
    )

    prompts = mock_prompt(
        json.dumps({"phenotypes": ["test"]}),  # phenotypes_all, only one text, so only once.
        json.dumps({"phenotypes": ["test"]}),  # phenotypes_observation, only one text, so only once.
        json.dumps({"phenotypes": []}),  # phenotypes_acronyms, only one text, so only once.
    )
    content_extractor = PromptBasedContentExtractor(
        list(fields.keys()),
        prompts,
        mock_observation([observation]),
        mock_phenotype_searcher([]),
        mock_phenotype_fetcher(),
    )
    content = content_extractor.extract(paper, fields["gene"])

    assert len(content) == 1
    for key in fields:
        assert content[0][key] == fields[key]


def test_prompt_based_content_extractor_phenotype_hpo_description(
    paper: Paper, mock_prompt: Any, mock_observation: Any, mock_phenotype_searcher: Any, mock_phenotype_fetcher: Any
) -> None:
    fields = {
        "gene": "CHI3L1",
        "paper_id": "12345678",
        "hgvs_c": "c.1234A>G",
        "paper_variant": "c.1234A>G",
        "individual_id": "unknown",
        "phenotype": "test (HP:012345)",
    }

    observation = Observation(
        variant=HGVSVariant(fields["hgvs_c"], fields["gene"], "transcript", True, True, None, None, []),
        individual=fields["individual_id"],
        texts=[TextSection("TEST", "test", 0, "Here is the observation text.", "unknown")],
        variant_descriptions=fields["paper_variant"].split(", "),
        patient_descriptions=[fields["individual_id"]],
        paper_id=fields["paper_id"],
    )

    prompts = mock_prompt(
        json.dumps({"phenotypes": ["test"]}),  # phenotypes_all, only one text, so only once.
        json.dumps({"phenotypes": ["test"]}),  # phenotypes_observation, only one text, so only once.
        json.dumps({"phenotypes": ["HP:012345"]}),  # phenotypes_acronyms, only one text, so only once.
    )

    phenotype_fetcher = mock_phenotype_fetcher({"id": "HP:012345", "name": "test"})
    content_extractor = PromptBasedContentExtractor(
        list(fields.keys()),
        prompts,
        mock_observation([observation]),
        mock_phenotype_searcher([]),
        phenotype_fetcher,
    )
    content = content_extractor.extract(paper, fields["gene"])

    assert len(content) == 1
    for key in fields:
        assert content[0][key] == fields[key]


def test_prompt_based_content_extractor_phenotype_empty_pheno_search(
    paper: Paper, mock_prompt: Any, mock_observation: Any, mock_phenotype_searcher: Any, mock_phenotype_fetcher: Any
) -> None:
    fields = {
        "gene": "CHI3L1",
        "paper_id": "12345678",
        "hgvs_c": "c.1234A>G",
        "paper_variant": "c.1234A>G",
        "individual_id": "unknown",
        "phenotype": "test",
    }

    observation = Observation(
        variant=HGVSVariant(fields["hgvs_c"], fields["gene"], "transcript", True, True, None, None, []),
        individual=fields["individual_id"],
        texts=[TextSection("TEST", "test", 0, "Here is the observation text.", "unknown")],
        variant_descriptions=fields["paper_variant"].split(", "),
        patient_descriptions=[fields["individual_id"]],
        paper_id=fields["paper_id"],
    )

    prompts = mock_prompt(
        json.dumps({"phenotypes": ["test"]}),  # phenotypes_all, only one text, so only once.
        json.dumps({"phenotypes": ["test"]}),  # phenotypes_observation, only one text, so only once.
        json.dumps({"phenotypes": ["test"]}),  # phenotypes_acronyms, only one text, so only once.
        json.dumps({}),  # phenotypes_simplify, only one text, so only once.
    )

    content_extractor = PromptBasedContentExtractor(
        list(fields.keys()),
        prompts,
        mock_observation([observation]),
        mock_phenotype_searcher([]),
        mock_phenotype_fetcher({}),
    )
    content = content_extractor.extract(paper, fields["gene"])

    assert len(content) == 1
    for key in fields:
        assert content[0][key] == fields[key]


def test_prompt_based_content_extractor_phenotype_simplification(
    paper: Paper, mock_prompt: Any, mock_observation: Any, mock_phenotype_searcher: Any, mock_phenotype_fetcher: Any
) -> None:
    fields = {
        "gene": "CHI3L1",
        "paper_id": "12345678",
        "hgvs_c": "c.1234A>G",
        "paper_variant": "c.1234A>G",
        "individual_id": "unknown",
        "phenotype": "test_simplified (HP:0321)",
    }

    observation = Observation(
        variant=HGVSVariant(fields["hgvs_c"], fields["gene"], "transcript", True, True, None, None, []),
        individual=fields["individual_id"],
        texts=[TextSection("TEST", "test", 0, "Here is the observation text.", "unknown")],
        variant_descriptions=fields["paper_variant"].split(", "),
        patient_descriptions=[fields["individual_id"]],
        paper_id=fields["paper_id"],
    )

    prompts = mock_prompt(
        json.dumps({"phenotypes": ["test"]}),  # phenotypes_all, only one text, so only once.
        json.dumps({"phenotypes": ["test"]}),  # phenotypes_observation, only one text, so only once.
        json.dumps({"phenotypes": ["test"]}),  # phenotypes_acronyms, only one text, so only once.
        json.dumps({}),  # phenotypes_candidates, initial value
        json.dumps({"simplified": ["test_simplified"]}),  # phenotypes_simplify, only one text, so only once.
        json.dumps({"match": "test_simplified (HP:0321)"}),  # phenotypes_candidates, simplified value
    )

    pheno_searcher = mock_phenotype_searcher(
        [{"id": "HP:0123", "name": "test", "definition": "test", "synonyms": "test"}],
        [{"id": "HP:0321", "name": "test_simplified", "definition": "test", "synonyms": "test"}],
    )
    content_extractor = PromptBasedContentExtractor(
        list(fields.keys()),
        prompts,
        mock_observation([observation]),
        pheno_searcher,
        mock_phenotype_fetcher({}),
    )
    content = content_extractor.extract(paper, fields["gene"])

    assert len(content) == 1
    for key in fields:
        assert content[0][key] == fields[key]


def test_prompt_based_content_extractor_phenotype_no_results_in_text(
    paper: Paper, mock_prompt: Any, mock_observation: Any, mock_phenotype_searcher: Any, mock_phenotype_fetcher: Any
) -> None:
    fields = {
        "gene": "CHI3L1",
        "paper_id": "12345678",
        "hgvs_c": "c.1234A>G",
        "paper_variant": "c.1234A>G",
        "individual_id": "unknown",
        "phenotype": "",
    }

    observation = Observation(
        variant=HGVSVariant(fields["hgvs_c"], fields["gene"], "transcript", True, True, None, None, []),
        individual=fields["individual_id"],
        texts=[TextSection("TEST", "test", 0, "Here is the observation text.", "unknown")],
        variant_descriptions=fields["paper_variant"].split(", "),
        patient_descriptions=[fields["individual_id"]],
        paper_id=fields["paper_id"],
    )

    prompts = mock_prompt(
        json.dumps({"phenotypes": []}),  # phenotypes_all, only one text, so only once.
    )

    content_extractor = PromptBasedContentExtractor(
        list(fields.keys()),
        prompts,
        mock_observation([observation]),
        mock_phenotype_searcher([]),
        mock_phenotype_fetcher({}),
    )
    content = content_extractor.extract(paper, fields["gene"])

    assert len(content) == 1
    for key in fields:
        assert content[0][key] == fields[key]


def test_prompt_based_content_extractor_phenotype_no_results_for_observation(
    paper: Paper, mock_prompt: Any, mock_observation: Any, mock_phenotype_searcher: Any, mock_phenotype_fetcher: Any
) -> None:
    fields = {
        "gene": "CHI3L1",
        "paper_id": "12345678",
        "hgvs_c": "c.1234A>G",
        "paper_variant": "c.1234A>G",
        "individual_id": "unknown",
        "phenotype": "",
    }

    observation = Observation(
        variant=HGVSVariant(fields["hgvs_c"], fields["gene"], "transcript", True, True, None, None, []),
        individual=fields["individual_id"],
        texts=[TextSection("TEST", "test", 0, "Here is the observation text.", "unknown")],
        variant_descriptions=fields["paper_variant"].split(", "),
        patient_descriptions=[fields["individual_id"]],
        paper_id=fields["paper_id"],
    )

    prompts = mock_prompt(
        json.dumps({"phenotypes": ["test"]}),  # phenotypes_all, only one text, so only once.
        json.dumps({"phenotypes": []}),  # phenotypes_obs, only one text, so only once.
    )

    content_extractor = PromptBasedContentExtractor(
        list(fields.keys()),
        prompts,
        mock_observation([observation]),
        mock_phenotype_searcher([]),
        mock_phenotype_fetcher({}),
    )
    content = content_extractor.extract(paper, fields["gene"])

    assert len(content) == 1
    for key in fields:
        assert content[0][key] == fields[key]


def test_prompt_based_content_extractor_phenotype_specific_individual(
    paper: Paper, mock_prompt: Any, mock_observation: Any, mock_phenotype_searcher: Any, mock_phenotype_fetcher: Any
) -> None:
    fields = {
        "gene": "CHI3L1",
        "paper_id": "12345678",
        "hgvs_c": "c.1234A>G",
        "paper_variant": "c.1234A>G",
        "individual_id": "proband",
        "phenotype": "",
    }

    observation = Observation(
        variant=HGVSVariant(fields["hgvs_c"], fields["gene"], "transcript", True, True, None, None, []),
        individual=fields["individual_id"],
        texts=[TextSection("TEST", "test", 0, "Here is the observation text.", "unknown")],
        variant_descriptions=fields["paper_variant"].split(", "),
        patient_descriptions=[fields["individual_id"]],
        paper_id=fields["paper_id"],
    )

    prompts = mock_prompt(
        json.dumps({"phenotypes": []}),  # phenotypes_all, only one text, so only once.
    )

    content_extractor = PromptBasedContentExtractor(
        list(fields.keys()),
        prompts,
        mock_observation([observation]),
        mock_phenotype_searcher([]),
        mock_phenotype_fetcher({}),
    )
    content = content_extractor.extract(paper, fields["gene"])

    assert len(content) == 1
    for key in fields:
        assert content[0][key] == fields[key]


def test_prompt_based_content_extractor_phenotype_table_texts(
    paper: Paper, mock_prompt: Any, mock_observation: Any, mock_phenotype_searcher: Any, mock_phenotype_fetcher: Any
) -> None:
    fields = {
        "gene": "CHI3L1",
        "paper_id": "12345678",
        "hgvs_c": "c.1234A>G",
        "paper_variant": "c.1234A>G",
        "individual_id": "unknown",
        "phenotype": "",
    }

    observation = Observation(
        variant=HGVSVariant(fields["hgvs_c"], fields["gene"], "transcript", True, True, None, None, []),
        individual=fields["individual_id"],
        texts=[
            TextSection("TEST", "test", 0, "Here is the observation text.", "unknown"),
            TextSection("TABLE", "table", 0, "Here is the table text.", "unknown"),
        ],
        variant_descriptions=fields["paper_variant"].split(", "),
        patient_descriptions=[fields["individual_id"]],
        paper_id=fields["paper_id"],
    )

    prompts = mock_prompt(
        json.dumps({"phenotypes": []}),  # phenotypes_all, two texts
        json.dumps({"phenotypes": []}),  # phenotypes_all
    )

    content_extractor = PromptBasedContentExtractor(
        list(fields.keys()),
        prompts,
        mock_observation([observation]),
        mock_phenotype_searcher([]),
        mock_phenotype_fetcher({}),
    )
    content = content_extractor.extract(paper, fields["gene"])

    assert prompts.call_count("prompt_file") == 2  # ensure both prompts were used.
    assert len(content) == 1
    for key in fields:
        assert content[0][key] == fields[key]


def test_prompt_based_content_extractor_json_prompt_response(
    paper: Paper, mock_prompt: Any, mock_observation: Any, mock_phenotype_searcher: Any, mock_phenotype_fetcher: Any
) -> None:
    fields = {
        "gene": "CHI3L1",
        "paper_id": "12345678",
        "hgvs_c": "c.1234A>G",
        "paper_variant": "c.1234A>G",
        "individual_id": "unknown",
        "zygosity": json.dumps({"zygosity": {"key": "value"}}),
    }

    observation = Observation(
        variant=HGVSVariant(fields["hgvs_c"], fields["gene"], "transcript", True, True, None, None, []),
        individual=fields["individual_id"],
        texts=[TextSection("TEST", "test", 0, "Here is the observation text.", "unknown")],
        variant_descriptions=fields["paper_variant"].split(", "),
        patient_descriptions=[fields["individual_id"]],
        paper_id=fields["paper_id"],
    )

    prompts = mock_prompt(fields["zygosity"])
    content_extractor = PromptBasedContentExtractor(
        list(fields.keys()), prompts, mock_observation([observation]), mock_phenotype_searcher, mock_phenotype_fetcher
    )
    content = content_extractor.extract(paper, fields["gene"])

    assert len(content) == 1
    assert content[0]["zygosity"] == '{"key": "value"}'


xmldoc = """
    <document>
        <id>7933980</id>
        {content}
    </document>
"""


def test_fulltext():
    xml1 = """
        <passage>
            <infon key="section_type">TITLE</infon>
            <infon key="type">front</infon>
            <offset>0</offset>
            <text>test
                title</text>
        </passage>
"""
    assert get_fulltext(None) == ""
    assert get_fulltext(xmldoc.format(content=xml1), include=["TITLE"]) == "test title"
    assert get_fulltext(xmldoc.format(content=xml1), exclude=["TITLE"]) == ""
    assert get_fulltext(xmldoc.format(content=xml1), include=["TITLE"], exclude=["TITLE"]) == ""


def test_fulltext_missing():
    xml1 = """
        <passage>
        </passage>
"""
    xml2 = """
        <passage>
            <infon key="section_type">TITLE</infon>
        </passage>
"""
    xml3 = """
        <passage>
            <infon key="section_type">TITLE</infon>
            <infon key="type">front</infon>
        </passage>
"""
    with pytest.raises(ValueError) as e:
        get_fulltext(xmldoc.format(content=xml1))
    assert str(e.value) == "Missing 'section_type' infon element in passage for document 7933980"
    with pytest.raises(ValueError) as e:
        get_fulltext(xmldoc.format(content=xml2))
    assert str(e.value) == "Missing 'type' infon element in passage for document 7933980"
    with pytest.raises(ValueError) as e:
        get_fulltext(xmldoc.format(content=xml3))
    assert str(e.value) == "Missing 'offset' infon element in TITLE passage for document 7933980"
