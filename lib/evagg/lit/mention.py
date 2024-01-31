import logging
import re
from typing import Any, Dict, List, Sequence, Set, Tuple

import Bio.SeqUtils

from lib.evagg.lit import IAnnotateEntities, IFindVariantMentions
from lib.evagg.ref import IGeneLookupClient
from lib.evagg.types import IPaperQuery, Paper

logger = logging.getLogger(__name__)


class VariantMentionFinder(IFindVariantMentions):
    def __init__(self, entity_annotator: IAnnotateEntities, gene_lookup_client: IGeneLookupClient) -> None:
        self._entity_annotator = entity_annotator
        self._gene_lookup_client = gene_lookup_client

    def _get_variant_ids(self, annotations: Dict[str, Any], query_gene_id: int, query_gene_symbol: str) -> Set[str]:
        variants_in_query_gene: Set[str] = set()

        if len(annotations) == 0:
            return variants_in_query_gene

        # TODO: filtering based on gene_id is probably too strict - the annotator may have identified a variant but
        # failed to associate it with a gene. This is particularly the case for novel variants if the annotator relies
        # on curated references for variant-gene association.

        for passage in annotations["passages"]:
            for annotation in passage["annotations"]:
                if annotation["infons"]["type"] == "Mutation":
                    if "gene_id" in annotation["infons"] and annotation["infons"]["gene_id"] == query_gene_id:
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
        query_gene_ids = self._gene_lookup_client.gene_id_for_symbol(*[v.gene for v in query.terms()])

        # Annotate entities in the paper.
        annotations = self._entity_annotator.annotate(paper)

        # Search all of the annotations for `Mutations` with the query gene ids.
        variant_gene_ids: Set[Tuple[str, int, str]] = set()
        for gene_symbol, gene_id in query_gene_ids.items():
            # Variant IDs are dictated by the annotator here. Using PubTator, they could be RSIDs, or GENE:hgvs_*
            # strings. This will cause problems when comparing to the truth set, which uses GENE:hgvs_c strings.
            # TODO, normalize nomenclature.
            variant_ids = self._get_variant_ids(annotations, gene_id, gene_symbol)
            variant_gene_tuples = {(gene_symbol, gene_id, v) for v in variant_ids}
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


class TruthsetVariantMentionFinder(IFindVariantMentions):
    def __init__(self, entity_annotator: IAnnotateEntities, gene_lookup_client: IGeneLookupClient) -> None:
        self._entity_annotator = entity_annotator
        self._gene_lookup_client = gene_lookup_client

    SHORTHAND_HGVS_PATTERN = re.compile(r"^p\.([A-Za-z])(\d+)([A-Za-z\*]|fs|del)$")
    LONGHAND_HGVS_PATTERN = re.compile(r"^p\.([A-Za-z]{3})(\d+)([A-Za-z]{3}|\*|fs|del)$")
    UNCONVERTED_AA_CODES = {"del", "fs"}

    def _generate_lowercase_hgvsp_representations(self, hgvsp_raw: str) -> List[str]:
        output = []

        match_short = re.match(self.SHORTHAND_HGVS_PATTERN, hgvsp_raw)
        if match_short:
            aa1, num, aa2 = match_short.groups()
            if str(aa1).isalpha():
                aa1 = Bio.SeqUtils.seq3(aa1.upper())
            if str(aa2).isalpha() and aa1 not in self.UNCONVERTED_AA_CODES:
                aa2 = Bio.SeqUtils.seq3(aa2.upper())
            output.append(f"p.{aa1}{num}{aa2}".lower())
            output.append(hgvsp_raw.lower())
            return output

        match_long = re.match(self.LONGHAND_HGVS_PATTERN, hgvsp_raw)
        if match_long:
            aa1, num, aa2 = match_long.groups()
            if str(aa1).isalpha():
                aa1 = Bio.SeqUtils.seq1(aa1.upper())
            if str(aa2).isalpha() and aa1 not in self.UNCONVERTED_AA_CODES:
                aa2 = Bio.SeqUtils.seq1(aa2.upper())
            output.append(hgvsp_raw.lower())
            output.append(f"p.{aa1}{num}{aa2}".lower())
            return output

        raise ValueError(f"Could not parse hgvs_p: {hgvsp_raw}")

    def _gather_mentions_for_variant(
        self, annotations: Dict[str, Any], gene_id: int, evidence: dict[str, str]
    ) -> Sequence[Dict[str, Any]]:
        mentions: List[Dict[str, Any]] = []

        if len(annotations) == 0:
            return mentions

        hgvsps = self._generate_lowercase_hgvsp_representations(evidence["hgvs_p"])

        # TODO, this approach to variant nomenclature is pretty brittle. It will only match variant mentions where the
        # hgvs_c or hgvs_p representation in a paper is exactly the same as what's in the truth set (case insensitive).
        # Thus expressions of the variant in paper text that don't conform to hgvs nomenclature standards might be
        # missed.
        for passage in annotations["passages"]:
            for annotation in passage["annotations"]:
                save = False
                if not annotation["infons"]["type"] == "Mutation":
                    continue
                if annotation["infons"]["subtype"] == "DNAMutation":
                    if "hgvs" in annotation["infons"] and annotation["infons"]["hgvs"] == evidence["hgvs_c"]:
                        save = True
                    elif (
                        annotation["infons"]["identifier"] == evidence["hgvs_c"]
                        or annotation["text"] == evidence["hgvs_c"]
                    ):
                        save = True
                if annotation["infons"]["subtype"] == "ProteinMutation":
                    if "hgvs" in annotation["infons"] and annotation["infons"]["hgvs"].lower() in hgvsps:
                        save = True
                    elif annotation["infons"]["identifier"] and annotation["infons"]["identifier"].lower() in hgvsps:
                        save = True
                    elif annotation["text"] and annotation["text"].lower() in hgvsps:
                        save = True
                if save:
                    to_add = passage.copy()
                    to_add.pop("annotations")
                    to_add["gene_symbol"] = evidence["gene"]
                    mentions.append(to_add)
                    break

        return mentions

    def find_mentions(self, query: IPaperQuery, paper: Paper) -> Dict[str, Sequence[Dict[str, Any]]]:
        # Get information on the variants that should be mentioned in the paper per the truth set and that are included
        # in the current query.
        query_genes = {v.gene for v in query.terms()}
        truth_rows = [d for d in paper.evidence.values() if d["gene"] in query_genes]

        # Get the gene ids.
        query_gene_ids = self._gene_lookup_client.gene_id_for_symbol(*[v["gene"] for v in truth_rows])

        # Get the annotated paper.
        annotations = self._entity_annotator.annotate(paper)

        # Search all of the annotations for `Mutations` that match the variants from the truth set.
        mentions: Dict[str, Sequence[Dict[str, Any]]] = {}
        for variant_dict in truth_rows:
            gene_symbol = variant_dict["gene"]
            if gene_symbol not in query_gene_ids:
                logger.warning("Gene symbol not found in query gene ids")
                continue

            identifier = f"{gene_symbol}:{variant_dict['hgvs_c']}"
            mentions[identifier] = self._gather_mentions_for_variant(
                annotations, query_gene_ids[gene_symbol], variant_dict
            )

        return mentions
