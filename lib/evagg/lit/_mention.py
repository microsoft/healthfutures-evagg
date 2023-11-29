from typing import Any, Dict, List, Sequence, Set, Tuple

from lib.evagg.lit import IAnnotateEntities, IFindVariantMentions
from lib.evagg.ref import INcbiGeneClient
from lib.evagg.types import IPaperQuery, Paper


class VariantMentionFinder(IFindVariantMentions):
    def __init__(self, entity_annotator: IAnnotateEntities, ncbi_gene_client: INcbiGeneClient) -> None:
        self._entity_annotator = entity_annotator
        self._ncbi_gene_client = ncbi_gene_client

    def _get_variant_ids(self, annotations: Dict[str, Any], query_gene_id: int) -> Set[str]:
        variants_in_query_gene: Set[str] = set()

        if len(annotations) == 0:
            return variants_in_query_gene

        # TODO: this is too strict - the annotator may have identified a variant but failed to associate it with a gene.
        # Specifically PubTator uses tmVar which often misses this annotation. As a result, we miss the variant.
        for passage in annotations["passages"]:
            for annotation in passage["annotations"]:
                if annotation["infons"]["type"] == "Mutation":
                    if "gene_id" in annotation["infons"] and annotation["infons"]["gene_id"] == query_gene_id:
                        # TODO: this not necessarily specific enough. The identifier may be a dbSNP id, or potentially
                        # just the HGVS id without the protein context. The former isn't specific to the actual variant
                        # (e.g., could have two alt alleles), the latter is specific to the gene (e.g., multiple genes)
                        # might have the same variant.
                        if annotation["infons"]["identifier"] is not None:
                            variants_in_query_gene.add(annotation["infons"]["identifier"])
        return variants_in_query_gene

    def _gather_mentions(self, annotations: Dict[str, Any], variant_id: str) -> Sequence[Dict[str, Any]]:
        mentions: List[Dict[str, Any]] = []

        if len(annotations) == 0:
            return mentions

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

    def find_mentions(self, query: IPaperQuery, paper: Paper) -> Dict[str, Sequence[Dict[str, Any]]]:
        # Get the gene_id(s) for the query gene(s).
        query_gene_ids = self._ncbi_gene_client.gene_id_for_symbol([v.gene for v in query.terms()])

        # Annotate entities in the paper.
        annotations = self._entity_annotator.annotate(paper)

        # Search all of the annotations for `Mutations` with the query gene ids.
        variant_gene_ids: Set[Tuple[str, int, str]] = set()
        for symbol, id in query_gene_ids.items():
            variant_ids = self._get_variant_ids(annotations, id)
            variant_gene_tuples = {(symbol, id, v) for v in variant_ids}
            variant_gene_ids.update(variant_gene_tuples)

        # Now collect all the text chunks that mention each variant.
        # Record the gene id for each variant as well.
        mentions: Dict[str, Sequence[Dict[str, Any]]] = {}
        for gene_symbol, gene_id, variant_id in variant_gene_ids:
            mentions[variant_id] = self._gather_mentions(annotations, variant_id)
            for m in mentions[variant_id]:
                m["gene_id"] = gene_id
                m["gene_symbol"] = gene_symbol

        return mentions
