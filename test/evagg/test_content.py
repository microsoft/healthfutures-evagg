import json
from typing import Any, Dict, Sequence

import pytest

from lib.evagg import SemanticKernelContentExtractor, SimpleContentExtractor
from lib.evagg.lit import IFindVariantMentions
from lib.evagg.ref import INcbiSnpClient
from lib.evagg.sk import ISemanticKernelClient
from lib.evagg.types import IPaperQuery, Paper, Query


def test_simple_content_extractor():
    paper = Paper(
        id="12345678", citation="Doe, J. et al. Test Journal 2021", abstract="This is a test paper.", pmcid="PMC123"
    )
    fields = ["gene", "hgvsp", "inheritance", "phenotype"]
    extractor = SimpleContentExtractor(fields)
    result = extractor.extract(paper, Query("CHI3L1:p.Y34C"))
    assert len(result) == 1
    assert result[0]["gene"] == "CHI3L1"
    assert result[0]["hgvsp"] == "p.Y34C"
    assert result[0]["inheritance"] == "AD"
    assert result[0]["phenotype"] == "Long face (HP:0000276)"


class MockMentionFinder(IFindVariantMentions):
    mentions: Dict[str, Sequence[Dict[str, Any]]] = {
        "var1": [{"text": "This is a test paper.", "gene_id": 1116, "gene_symbol": "CHI3L1"}],
        "var2": [
            {"text": "This is another test paper.", "gene_id": 1116, "gene_symbol": "CHI3L1"},
        ],
    }

    def find_mentions(self, query: IPaperQuery, paper: Paper) -> Dict[str, Sequence[Dict[str, Any]]]:
        return self.mentions


class MockSemanticKernelClient(ISemanticKernelClient):
    def run_completion_function(self, skill: str, function: str, context_variables: Any) -> str:
        return json.dumps({function: "test"})


class MockNcbiSnpClient(INcbiSnpClient):
    def __init__(self, hgvsc: str, hgvsp: str) -> None:
        self._hgvsc = hgvsc
        self._hgvsp = hgvsp

    def hgvs_from_rsid(self, rsid: str) -> Dict[str, str | None]:
        return {
            "hgvsc": self._hgvsc,
            "hgvsp": self._hgvsp,
        }


def test_sk_content_extractor_valid_fields():
    fields = {
        "gene": "CHI3L1",
        "paper_id": "12345678",
        "hgvsc": "c.A100G",
        "hgvsp": "g.Py34C",
        "phenotype": "test",
        "zygosity": "test",
        "inheritance": "test",
    }

    content_extractor = SemanticKernelContentExtractor(
        list(fields.keys()),
        MockSemanticKernelClient(),
        MockMentionFinder(),
        MockNcbiSnpClient(hgvsc=fields["hgvsc"], hgvsp=fields["hgvsp"]),
    )
    paper = Paper(id=fields["paper_id"], citation="citation", abstract="This is a test paper.", pmcid="PMC123")
    content = content_extractor.extract(paper, Query(f"{fields['gene']}:p.Y34C"))

    print(content)
    # Based on the above, there should be two elements in `content`, each containing a dict keyed by fields, with a
    # fake value.
    for k in MockMentionFinder().mentions.keys():
        row = [c for c in content if c["variant"] == k]
        assert len(row) == 1
        row = row[0]

        for f in fields:
            assert f in row.keys()
            assert row[f] == fields[f]


def test_sk_content_extractor_invalid_fields():
    fields = ["not a field"]

    content_extractor = SemanticKernelContentExtractor(
        fields,
        MockSemanticKernelClient(),
        MockMentionFinder(),
        MockNcbiSnpClient(hgvsc="irrelevant", hgvsp="irrelevant"),
    )
    paper = Paper(id="12345678", citation="citation", abstract="This is a test paper.", pmcid="PMC123")

    with pytest.raises(ValueError):
        content_extractor.extract(paper, Query("CHI3L1:p.Y34C"))
