from typing import Any, Dict, Sequence, Tuple

from lib.evagg.lit import IAnnotateEntities, VariantMentionFinder
from lib.evagg.types import MultiQuery, Paper, Query

GENE_ID_PAIRS = {
    "COQ2": 27235,
    "CHI3L1": 1116,
    "PRKCG": 5582,
}


class MockAnnotator(IAnnotateEntities):
    def __init__(self, gene_id_variant_tuples: Sequence[Tuple[int, str]]) -> None:
        self._gene_id_variant_tuples = gene_id_variant_tuples

    def annotate(self, paper: Paper) -> Dict[str, Any]:
        passages = {
            "passages": [
                {
                    "text": "Here is the passage text.",
                    "annotations": [
                        {
                            "infons": {
                                "type": "Mutation",
                                "identifier": variant,
                                "gene_id": gene_id,
                            }
                        }
                        for gene_id, variant in self._gene_id_variant_tuples
                    ],
                }
            ]
        }
        return passages


def test_find_single_mention_single_query():
    gene_symbol = list(GENE_ID_PAIRS.keys())[0]
    gene_id = GENE_ID_PAIRS[gene_symbol]
    query = Query(gene_symbol, "varX")

    mock_annotator = MockAnnotator([(gene_id, "var1")])
    finder = VariantMentionFinder(mock_annotator)

    paper = Paper(id="123")

    mentions = finder.find_mentions(query, paper)

    # There should be one mention, corresponding to var1.
    assert len(mentions) == 1
    assert "var1" in mentions
    # This mention should reference one passage.
    assert len(mentions["var1"]) == 1
    assert mentions["var1"][0]["gene_id"] == gene_id
    assert mentions["var1"][0]["gene_symbol"] == gene_symbol


def test_find_multiple_mentions_single_query():
    gene_symbol = list(GENE_ID_PAIRS.keys())[0]
    gene_id = GENE_ID_PAIRS[gene_symbol]
    query = Query(gene_symbol, "varX")

    mock_annotator = MockAnnotator([(gene_id, "var1"), (gene_id, "var2")])
    finder = VariantMentionFinder(mock_annotator)

    paper = Paper(id="123")

    mentions = finder.find_mentions(query, paper)

    # There should be two mentions, one corresponding to var1, another corresponding to var2.
    assert len(mentions) == 2
    assert "var1" in mentions
    assert "var2" in mentions
    # Each mention should reference one passage.
    for varname in ["var1", "var2"]:
        assert len(mentions[varname]) == 1
        assert mentions[varname][0]["gene_id"] == gene_id
        assert mentions[varname][0]["gene_symbol"] == gene_symbol


def test_find_multiple_mentions_multi_query():
    gene_symbols = list(GENE_ID_PAIRS.keys())
    gene_ids = list(GENE_ID_PAIRS.values())

    query = MultiQuery([f"{gene_symbol}:varX" for gene_symbol in gene_symbols])

    # Ensure that variant ids are unique across genes.
    mock_annotator = MockAnnotator([(gene_id, f"var{gene_id}") for gene_id in gene_ids])
    finder = VariantMentionFinder(mock_annotator)

    paper = Paper(id="123")

    mentions = finder.find_mentions(query, paper)

    # There should be one mention per query gene.
    assert len(mentions) == len(gene_symbols)
    for gene_symbol, gene_id in GENE_ID_PAIRS.items():
        assert f"var{gene_id}" in mentions
        assert len(mentions[f"var{gene_id}"]) == 1
        assert mentions[f"var{gene_id}"][0]["gene_id"] == gene_id
        assert mentions[f"var{gene_id}"][0]["gene_symbol"] == gene_symbol


def test_find_no_mentions():
    gene_symbol = list(GENE_ID_PAIRS.keys())[0]
    gene_id = GENE_ID_PAIRS[gene_symbol]

    assert gene_symbol != "FAM111B"  # Just to be sure.
    query = Query("FAM111B", "varX")

    mock_annotator = MockAnnotator([(gene_id, "var1")])
    finder = VariantMentionFinder(mock_annotator)

    paper = Paper(id="123")

    mentions = finder.find_mentions(query, paper)

    # There should be no mentions.
    assert len(mentions) == 0
