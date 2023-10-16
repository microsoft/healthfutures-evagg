from typing import Any, List, Sequence, Set, Tuple

import requests

from lib.evagg.lit import IAnnotateEntities, IFindVariantMentions
from lib.evagg.types import IPaperQuery, Paper


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

    def find_mentions(self, query: IPaperQuery, paper: Paper) -> dict[str, Sequence[dict[str, Any]]]:
        # Get the gene_id for the query gene.
        # TODO, move this to a separate module/package.
        query_gene_ids = self._gene_id_for_symbol([v.gene for v in query.terms()])

        # Annotate entities in the paper.
        annotations = self._entity_annotator.annotate(paper)

        # Search all of the annotations for `Mutations` with the query gene ids.
        variant_gene_ids: Set[Tuple[str, str]] = set()
        for id in query_gene_ids:
            variant_ids = self._get_variant_ids(annotations, id)
            variant_gene_pairs = {(v, id) for v in variant_ids}
            variant_gene_ids.update(variant_gene_pairs)

        # Now collect all the text chunks that mention each variant.
        # Record the gene id for each variant as well.
        mentions: dict[str, Sequence[dict[str, Any]]] = {}
        for variant_id, gene_id in variant_gene_ids:
            # TODO, there's an error mode here where two genes in the query posess the same variant name.
            if variant_id in mentions:
                raise ValueError(f"Variant {variant_id} mentioned in multiple query genes.")

            mentions[variant_id] = self._gather_mentions(annotations, variant_id)
            for m in mentions[variant_id]:
                m["gene_id"] = gene_id

        return mentions
