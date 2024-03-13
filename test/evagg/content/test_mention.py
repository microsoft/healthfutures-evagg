from typing import Any, Dict, Sequence, Tuple

from lib.evagg.content import VariantMentionFinder
from lib.evagg.ref import IAnnotateEntities, IGeneLookupClient
from lib.evagg.types import HGVSVariant, ICreateVariants, Paper

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
                                "type": "Variant",
                                "identifier": variant,
                                "name": variant,
                                "gene_id": gene_id,
                            }
                        }
                        for gene_id, variant in self._gene_id_variant_tuples
                    ],
                }
            ]
        }
        return passages


class MockGeneLookupClient(IGeneLookupClient):
    def __init__(self, symbol_gene_id_tuples: Sequence[Tuple[str, int]]) -> None:
        self._gene_id_variant_tuples = symbol_gene_id_tuples

    def gene_id_for_symbol(self, *symbols: str, allow_synonyms: bool = False) -> Dict[str, int]:
        return {symbol: gene_id for symbol, gene_id in self._gene_id_variant_tuples if symbol in symbols}


class MockVariantFactory(ICreateVariants):
    def __init__(self) -> None:
        pass

    def parse(self, text_desc: str, gene_symbol: str | None, refseq: str | None = None) -> HGVSVariant:
        return HGVSVariant(text_desc, gene_symbol, "transcript", True, True)


def test_find_single_mention_single_query() -> None:
    gene_symbol = list(GENE_ID_PAIRS.keys())[0]
    gene_id = GENE_ID_PAIRS[gene_symbol]

    mock_annotator = MockAnnotator([(gene_id, "var1")])
    mock_gene_lookup_client = MockGeneLookupClient([(gene_symbol, gene_id)])
    mock_variant_factory = MockVariantFactory()
    finder = VariantMentionFinder(mock_annotator, mock_gene_lookup_client, mock_variant_factory)

    paper = Paper(id="123")

    mentions = finder.find_mentions(gene_symbol, paper)

    # There should be one mention, corresponding to var1.
    assert len(mentions) == 1
    vkey = HGVSVariant("var1", gene_symbol, "transcript", True, True)
    assert vkey in mentions
    # This mention should reference one passage.
    assert len(mentions[vkey]) == 1
    assert mentions[vkey][0]["gene_id"] == gene_id
    assert mentions[vkey][0]["gene_symbol"] == gene_symbol


def test_find_multiple_mentions_single_query() -> None:
    gene_symbol = list(GENE_ID_PAIRS.keys())[0]
    gene_id = GENE_ID_PAIRS[gene_symbol]

    mock_annotator = MockAnnotator([(gene_id, "var1"), (gene_id, "var2")])
    mock_gene_lookup_client = MockGeneLookupClient([(gene_symbol, gene_id)])
    mock_variant_factory = MockVariantFactory()
    finder = VariantMentionFinder(mock_annotator, mock_gene_lookup_client, mock_variant_factory)

    paper = Paper(id="123")

    mentions = finder.find_mentions(gene_symbol, paper)

    # There should be two mentions, one corresponding to var1, another corresponding to var2.
    assert len(mentions) == 2
    vkey1 = HGVSVariant("var1", gene_symbol, "transcript", True, True)
    vkey2 = HGVSVariant("var2", gene_symbol, "transcript", True, True)
    assert vkey1 in mentions
    assert vkey2 in mentions
    # Each mention should reference one passage.
    for variant in [vkey1, vkey2]:
        assert len(mentions[variant]) == 1
        assert mentions[variant][0]["gene_id"] == gene_id
        assert mentions[variant][0]["gene_symbol"] == gene_symbol


def test_find_no_mentions() -> None:
    gene_symbol = list(GENE_ID_PAIRS.keys())[0]
    gene_id = GENE_ID_PAIRS[gene_symbol]

    assert gene_symbol != "FAM111B"  # Just to be sure.
    query = "FAM111B"

    mock_annotator = MockAnnotator([(gene_id, "var1")])
    mock_gene_lookup_client = MockGeneLookupClient([(gene_symbol, gene_id)])
    mock_variant_factory = MockVariantFactory()
    finder = VariantMentionFinder(mock_annotator, mock_gene_lookup_client, mock_variant_factory)

    paper = Paper(id="123")

    mentions = finder.find_mentions(query, paper)

    # There should be no mentions.
    assert len(mentions) == 0
