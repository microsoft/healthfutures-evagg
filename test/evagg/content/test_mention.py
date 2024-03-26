from typing import Any, Dict, Sequence, Tuple

from lib.evagg.content.mention import HGVSVariantComparator, HGVSVariantFactory
from lib.evagg.ref import IAnnotateEntities, IGeneLookupClient
from lib.evagg.types import HGVSVariant, ICreateVariants, Paper

# GENE_ID_PAIRS = {
#     "COQ2": 27235,
#     "CHI3L1": 1116,
#     "PRKCG": 5582,
# }


# class MockAnnotator(IAnnotateEntities):
#     def __init__(self, gene_id_variant_tuples: Sequence[Tuple[int, str]]) -> None:
#         self._gene_id_variant_tuples = gene_id_variant_tuples

#     def annotate(self, paper: Paper) -> Dict[str, Any]:
#         passages = {
#             "passages": [
#                 {
#                     "text": "Here is the passage text.",
#                     "annotations": [
#                         {
#                             "infons": {
#                                 "type": "Variant",
#                                 "identifier": variant,
#                                 "name": variant,
#                                 "gene_id": gene_id,
#                             }
#                         }
#                         for gene_id, variant in self._gene_id_variant_tuples
#                     ],
#                 }
#             ]
#         }
#         return passages


# class MockGeneLookupClient(IGeneLookupClient):
#     def __init__(self, symbol_gene_id_tuples: Sequence[Tuple[str, int]]) -> None:
#         self._gene_id_variant_tuples = symbol_gene_id_tuples

#     def gene_id_for_symbol(self, *symbols: str, allow_synonyms: bool = False) -> Dict[str, int]:
#         return {symbol: gene_id for symbol, gene_id in self._gene_id_variant_tuples if symbol in symbols}


# class MockVariantFactory(ICreateVariants):
#     def __init__(self) -> None:
#         pass

#     def parse(
#         self,
#         text_desc: str,
#         gene_symbol: str | None,
#         refseq: str | None = None,
#         protein_consequence: HGVSVariant | None = None,
#     ) -> HGVSVariant:
#         return HGVSVariant(text_desc, gene_symbol, "transcript", True, True, protein_consequence)

#     gene_symbol = list(GENE_ID_PAIRS.keys())[0]
#     gene_id = GENE_ID_PAIRS[gene_symbol]

#     assert gene_symbol != "FAM111B"  # Just to be sure.
#     query = "FAM111B"

#     mock_annotator = MockAnnotator([(gene_id, "var1")])
#     mock_gene_lookup_client = MockGeneLookupClient([(gene_symbol, gene_id)])
#     mock_variant_factory = MockVariantFactory()
#     finder = VariantMentionFinder(mock_annotator, mock_gene_lookup_client, mock_variant_factory)

#     paper = Paper(id="123")

#     mentions = finder.find_mentions(query, paper)

#     # There should be no mentions.
#     assert len(mentions) == 0


def test_compare() -> None:
    comparator = HGVSVariantComparator()

    # Test two equivalent variants
    v1 = HGVSVariant("c.123A>G", "COQ2", None, True, True, None)
    assert comparator.compare(v1, v1) is v1
    v1 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.7)", True, True, None)
    assert comparator.compare(v1, v1) is v1

    # Test different variants
    v1 = HGVSVariant("c.123A>G", "COQ2", None, True, True, None)
    v2 = HGVSVariant("c.321A>G", "COQ2", None, True, True, None)
    assert comparator.compare(v1, v2) is None
    v1 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.7)", True, True, None)
    v2 = HGVSVariant("c.321A>G", "COQ2", "NP_123456.1(NM_123456.7)", True, True, None)
    assert comparator.compare(v1, v2) is None

    # Test two variants with different refseqs.
    v1 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.7)", True, True, None)
    v2 = HGVSVariant("c.123A>G", "COQ2", "NM_123456.8", True, True, None)
    assert comparator.compare(v1, v2) is v1
    v2 = HGVSVariant("c.123A>G", "COQ2", "NM_654321.8", True, True, None)
    assert comparator.compare(v1, v2) is None
    assert comparator.compare(v1, v2, True) is v1

    # Test variants with different completeness.
    v1 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.7)", True, True, None)
    v2 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.7)", False, True, None)
    assert comparator.compare(v1, v2) is v1
    v3 = HGVSVariant("c.123A>G", "COQ2", "NM_123456.7", False, True, None)
    assert comparator.compare(v2, v3) is v2

    # Test variants with equal completeness but different refseq version numbers.
    v1 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1", True, True, None)
    v2 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.2", True, True, None)
    assert comparator.compare(v1, v2) is v2
    v2 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.0", True, True, None)
    assert comparator.compare(v1, v2) is v1

    v1 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.7)", True, True, None)
    v2 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.6)", True, True, None)
    assert comparator.compare(v1, v2) is v1
    v2 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_123456.8)", True, True, None)
    assert comparator.compare(v1, v2) is v2

    # Test weird edge cases.
    # V1 as a fallback because at least one refseq is shared, but there are not different
    # version numbers for the shared refseqs
    v2 = HGVSVariant("c.123A>G", "COQ2", "NP_123456.1(NM_654321.8)", True, True, None)
    assert comparator.compare(v1, v2) is v1


def test_compare_via_protein_consequence() -> None:
    v1 = HGVSVariant("c.123A>G", "COQ2", "NM_123456.7", True, True, None)
    v2 = HGVSVariant("p.Ala123Gly", "COQ2", "NP_123456.1", True, True, v1)
    comparator = HGVSVariantComparator()
    assert comparator.compare(v1, v2) is v1
    assert comparator.compare(v2, v1) is v1
