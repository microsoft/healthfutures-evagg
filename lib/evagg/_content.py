from typing import Any, List, Protocol, Sequence, Set

import requests

from lib.evagg.lit import IAnnotateEntities
from lib.evagg.sk import SemanticKernelClient

from ._base import Paper, Query
from ._interfaces import IExtractFields


class SimpleContentExtractor(IExtractFields):
    def __init__(self, fields: Sequence[str]) -> None:
        self._fields = fields

    def _field_to_value(self, field: str) -> str:
        if field == "gene":
            return "CHI3L1"
        if field == "variant":
            return "p.Y34C"
        if field == "MOI":
            return "AD"
        if field == "phenotype":
            return "Long face (HP:0000276)"
        if field == "functional data":
            return "No"
        else:
            return "Unknown"

    def extract(self, query: Query, paper: Paper) -> Sequence[dict[str, str]]:
        # Dummy implementation that returns a single variant with a static set of fields.
        return [{field: self._field_to_value(field) for field in self._fields}]


class IFindVariantMentions(Protocol):
    def find_mentions(self, query: Query, paper: Paper) -> dict[str, Sequence[dict[str, Any]]]:
        """Find variant mentions relevant to query that are mentioned in `paper`.

        Returns a dictionary mapping each variant to a list of text chunks that mention it.
        """
        ...


class VariantMentionFinder(IFindVariantMentions):
    def __init__(self, entity_annotator: IAnnotateEntities) -> None:
        self._entity_annotator = entity_annotator

    def _get_variant_ids(self, annotations: dict[str, Any], query_gene_id: str) -> Set[str]:
        variants_in_query_gene: Set[str] = set()

        for passage in annotations["passages"]:
            for annotation in passage["annotations"]:
                if annotation["infons"]["type"] == "Mutation":
                    if "gene_id" in annotation["infons"] and annotation["infons"]["gene_id"] == query_gene_id:
                        variants_in_query_gene.add(annotation["infons"]["identifier"])
        return variants_in_query_gene

    def _gene_id_for_symbol(self, symbols: Sequence[str]) -> dict[str, str]:
        # TODO, wrap in Bio.Entrez library as they're better about rate limiting and such.
        url = f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha/gene/symbol/{','.join(symbols)}/taxon/Human"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return {g["gene"]["symbol"]: g["gene"]["gene_id"] for g in r.json()["reports"]}

    def _gather_mentions(self, annotations: dict[str, Any], variant_id: str) -> Sequence[dict[str, Any]]:
        mentions: List[dict[str, Any]] = []

        for passage in annotations["passages"]:
            for annotation in passage["annotations"]:
                if (
                    annotation["infons"]["type"] == "Mutation"
                    and "identifier" in annotation["infons"]
                    and annotation["infons"]["identifier"] == variant_id
                ):
                    to_add = passage.copy()
                    to_add.pop("annotations")
                    mentions.append(to_add)
                    break

        return mentions

    def find_mentions(self, query: Query, paper: Paper) -> dict[str, Sequence[dict[str, Any]]]:
        # Get the gene_id for the query gene.
        # TODO, query member is private.
        # TODO, move this to a separate library.
        query_gene_id = self._gene_id_for_symbol([query._gene])[query._gene]

        # Annotate entities in the paper.
        annotations = self._entity_annotator.annotate(paper)

        # Search all of the annotations for `Mutations` with the query gene id.
        variant_ids = self._get_variant_ids(annotations, query_gene_id)

        # Now collect all the text chunks that mention each variant.
        mentions: dict[str, Sequence[dict[str, Any]]] = {}
        for variant_id in variant_ids:
            mentions[variant_id] = self._gather_mentions(annotations, variant_id)

        return mentions


class SemanticKernelContentExtractor(IExtractFields):
    def __init__(
        self, fields: Sequence[str], sk_client: SemanticKernelClient, mention_finder: IFindVariantMentions
    ) -> None:
        self._fields = fields
        self._sk_client = sk_client
        self._mention_finder = mention_finder

    def _excerpt_from_mentions(self, mentions: Sequence[dict[str, Any]]) -> str:
        return "\n\n".join([m["text"] for m in mentions])

    def extract(self, query: Query, paper: Paper) -> Sequence[dict[str, str]]:
        # Find all the variant mentions in the paper
        variants_mentions = self._mention_finder.find_mentions(query, paper)

        # For each variant/field pair, run the appropriate prompt.
        results: List[dict[str, str]] = []

        for variant in variants_mentions:
            variant_results: dict[str, str] = {}

            # Simplest thing we can think of.
            paper_excerpts = self._excerpt_from_mentions(variants_mentions[variant])
            context_variables = {
                "input": paper_excerpts,
                "variant": variant,
                "gene": query._gene,
            }
            for field in self._fields:
                result = self._sk_client.run_completion_function(
                    skill="content", function=field, context_variables=context_variables
                )
                variant_results[field] = result
            results.append(variant_results)
        return results
