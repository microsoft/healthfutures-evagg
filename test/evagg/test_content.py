import json
from typing import Any, Sequence

import pytest

from lib.evagg import SemanticKernelContentExtractor, SimpleContentExtractor
from lib.evagg.lit import IFindVariantMentions
from lib.evagg.sk import ISemanticKernelClient
from lib.evagg.types import IPaperQuery, Paper, Query


def test_simple_content_extractor():
    paper = Paper(
        id="12345678", citation="Doe, J. et al. Test Journal 2021", abstract="This is a test paper.", pmcid="PMC123"
    )
    fields = ["gene", "variant", "MOI", "phenotype", "functional data"]
    extractor = SimpleContentExtractor(fields)
    result = extractor.extract(paper, Query("CHI3L1", "p.Y34C"))
    assert len(result) == 1
    assert result[0]["gene"] == "CHI3L1"
    assert result[0]["variant"] == "p.Y34C"
    assert result[0]["MOI"] == "AD"
    assert result[0]["phenotype"] == "Long face (HP:0000276)"
    assert result[0]["functional data"] == "No"


class MockMentionFinder(IFindVariantMentions):
    mentions: dict[str, Sequence[dict[str, Any]]] = {
        "var1": [{"text": "This is a test paper.", "gene_id": 1116, "gene_symbol": "CHI3L1"}],
        "var2": [
            {"text": "This is another test paper.", "gene_id": 1116, "gene_symbol": "CHI3L1"},
        ],
    }

    def find_mentions(self, query: IPaperQuery, paper: Paper) -> dict[str, Sequence[dict[str, Any]]]:
        return self.mentions


class MockSemanticKernelClient(ISemanticKernelClient):
    def run_completion_function(self, skill: str, function: str, context_variables: Any) -> str:
        return json.dumps({function: "test"})


def test_sk_content_extractor_valid_fields():
    fields = {
        "gene": "CHI3L1",
        "paper_id": "12345678",
        "hgvsc": "",  # value derived from MockMentionFinder
        "hgvsp": "",  # value derived from MockMentionFinder
        "phenotype": "test",
        "zygosity": "test",
        "inheritance": "test",
    }

    content_extractor = SemanticKernelContentExtractor(
        list(fields.keys()), MockSemanticKernelClient(), MockMentionFinder()
    )
    paper = Paper(id=fields["paper_id"], citation="citation", abstract="This is a test paper.", pmcid="PMC123")
    content = content_extractor.extract(paper, Query(fields["gene"], "p.Y34C"))

    print(content)
    # Based on the above, there should be two elements in `content`, each containing a dict keyed by fields, with a
    # fake value.
    for k in MockMentionFinder().mentions.keys():
        row = [c for c in content if c["variant"] == k]
        assert len(row) == 1
        row = row[0]

        for f in fields:
            assert f in row.keys()
            if f in ["hgvsc", "hgvsp"]:
                assert row[f] == k
            else:
                assert row[f] == fields[f]


def test_sk_content_extractor_invalid_fields():
    fields = ["not a field"]

    content_extractor = SemanticKernelContentExtractor(fields, MockSemanticKernelClient(), MockMentionFinder())
    paper = Paper(id="12345678", citation="citation", abstract="This is a test paper.", pmcid="PMC123")

    with pytest.raises(ValueError):
        content_extractor.extract(paper, Query("CHI3L1", "p.Y34C"))
