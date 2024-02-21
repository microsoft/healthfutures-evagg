import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Set, Tuple

import Bio.SeqUtils

from lib.evagg.ref import (
    IAnnotateEntities,
    IBackTranslateVariants,
    IGeneLookupClient,
    INormalizeVariants,
    IRefSeqLookupClient,
)
from lib.evagg.types import HGVSVariant, ICreateVariants, Paper

from .interfaces import IFindVariantMentions

logger = logging.getLogger(__name__)


class HGVSVariantFactory(ICreateVariants):
    _normalizer: INormalizeVariants
    _back_translator: IBackTranslateVariants
    _refseq_client: IRefSeqLookupClient

    MITO_REFSEQ = "NC_012920.1"

    def __init__(
        self,
        normalizer: INormalizeVariants,
        back_translator: IBackTranslateVariants,
        refseq_client: IRefSeqLookupClient,
    ) -> None:
        self._normalizer = normalizer
        self._back_translator = back_translator
        self._refseq_client = refseq_client

    def _predict_refseq(self, text_desc: str, gene_symbol: str | None) -> str | None:
        """Predict the RefSeq for a variant based on its description and gene symbol."""
        if text_desc.startswith("p.") and gene_symbol:
            return self._refseq_client.protein_accession_for_symbol(gene_symbol)
        elif text_desc.startswith("c.") and gene_symbol:
            return self._refseq_client.transcript_accession_for_symbol(gene_symbol)
        elif text_desc.startswith("m."):
            # TODO: consider moving?
            return self.MITO_REFSEQ
        elif text_desc.startswith("g."):
            raise ValueError(f"Genomic (g. prefixed) variants must have a RefSeq. None was provided for {text_desc}")
        else:
            logger.warning(f"Unsupported HGVS type: {text_desc} with gene symbol {gene_symbol}")
            return None

    def try_parse(self, text_desc: str, gene_symbol: str | None, refseq: str | None = None) -> HGVSVariant:
        """Attempt to parse a variant based on description and an optional gene symbol and optional refseq.

        `gene_symbol` is required for protein (p.) and coding (c.) variants, but not mitochondrial (m.) or genomic (g.)
        variants.

        If not provided, `refseq` will be predicted based on the variant description and gene symbol for protein and
        coding variants. `refseq` is required for genomic (g.) variants.

        Raises a ValueError if the above requirements are not met, or if anything but (g., c., p., m.) is provided as
        the description prefix.

        Raises a ValueError if the variant description is not syntactically valid according to HGVS nomenclature.

        Raises a ValueError if the refseq and variant description are not compatible (e.g., a protein variant on a
        transcript refseq).
        """
        # If no refseq is provided, we'll try to predict it.
        if not refseq:
            refseq = self._predict_refseq(text_desc, gene_symbol)
            refseq_predicted = True
        else:
            refseq_predicted = False

        if not refseq:
            raise ValueError(
                f"RefSeq required for all variants. None was provided or predicted for {text_desc, gene_symbol}"
            )

        # Validate the variant for both syntactic and biological correctness.
        is_valid = self._validate_variant(refseq, text_desc)

        return HGVSVariant(
            hgvs_desc=text_desc,
            gene_symbol=gene_symbol,
            refseq=refseq,
            refseq_predicted=refseq_predicted,
            valid=is_valid,
        )

    def _validate_variant(self, refseq: str, text_desc: str) -> bool:
        # Validate the variant. Returns True if the variant is valid, False otherwise.
        return bool(self._normalizer.normalize(f"{refseq}:{text_desc}"))


@dataclass(frozen=True)
class HGVSVariantMention:
    """A representation of a text mention of a genetic variant in a paper."""

    text: str
    context: str  # TODO: Maybe should be a reference to a Passage object?
    variant: HGVSVariant


class VariantTopic:
    """A Representation of a topic in a paper pertaining to a genetic variant."""

    _variant_mentions: List[HGVSVariantMention]
    _variants: Set[HGVSVariant]

    _normalizer: INormalizeVariants
    _back_translator: IBackTranslateVariants

    def __init__(
        self,
        mentions: Sequence[HGVSVariantMention],
        normalizer: INormalizeVariants,
        back_translator: IBackTranslateVariants,
    ) -> None:
        self._variant_mentions = []
        self._variants = set()

        self._normalizer = normalizer
        self._back_translator = back_translator

        for mention in mentions:
            self.add_mention(mention)

    def __str__(self) -> str:
        return f"VariantTopic: {', '.join(str(v) for v in self._variants)}"

    def get_mentions(self) -> Sequence[HGVSVariantMention]:
        """Get all variant mentions in the topic."""
        return self._variant_mentions

    def add_mention(self, mention: HGVSVariantMention) -> None:
        """Add a mention to the topic."""
        # Add the mention to the list of mentions.
        self._variant_mentions.append(mention)

        # Additionally, add the variant to the set of variants if it's not already there.
        self._variants.add(mention.variant)

    def get_variants(self) -> List[HGVSVariant]:
        """Get all variants mentioned in the topic."""
        return list(self._variants)

    def _linked(self, a: HGVSVariant, b: HGVSVariant) -> bool:
        """Determine if two variants are linked.

        Variant mentions are linked if the variants they represent are biologically related. Two mentions are linked
        if any of the following conditions ar true:
        - `a` and `b` are equivalent variants on the same reference (e.g., same transcript)
        - `a` and `b` are equivalent variants on different references (e.g., transcript and genome)
        - `a` is the protein product of `b` or vice versa

        If either variant mention is invalid it can only be linked to another variant mention if they are equivalent.

        """
        # Handle equivalent variants.
        if a == b:
            return True

        # If either variant is invalid, they can only be linked if they're equivalent.
        if not a.valid or not b.valid:
            return False

        # If they're both in the same coordinate system (g, c, or p) and they're not equivalent, they aren't linked.
        if a.hgvs_desc[0] == b.hgvs_desc[0]:
            return False

        # If they're in different coordinate systems things get a little more complicated. We need to map and/or
        # translate to a common coordinate system to determine if they're linked.
        # Easiest thing to do is to map back to genomic coordinates and compare there.
        def c_to_g(c: str) -> List[str]:
            norm = self._normalizer.normalize(c)
            if not norm or "chromosomal_descriptions" not in norm:
                return []
            for desc in norm["chromosomal_descriptions"]:
                if desc["assembly"] == "GRCH38":
                    return [desc["g"]]
            return []

        def p_to_g(p: str) -> List[str]:
            txs = self._back_translator.back_translate(p)
            return [g for tx in txs for g in c_to_g(tx.replace("(", "").replace(")", ""))]

        def map_to_g(v: HGVSVariant) -> List[str]:
            if v.hgvs_desc.startswith("g."):
                return [str(v)]
            elif v.hgvs_desc.startswith("c."):
                return c_to_g(str(v))
            elif v.hgvs_desc.startswith("p."):
                return p_to_g(str(v))
            else:
                return []

        # If the intersection of the genomic mappings is non-empty, the variants are linked.
        # TODO, handle parenthetical reference sequences and predicted descriptions.
        return bool(set(map_to_g(a)) & set(map_to_g(b)))

    def match(self, other: HGVSVariantMention) -> bool:
        """Determine if the new variant mention is a topical match for the topic."""
        for variant in self._variants:
            if self._linked(variant, other.variant):
                return True
        return False


class VariantMentionFinder(IFindVariantMentions):
    def __init__(
        self,
        entity_annotator: IAnnotateEntities,
        gene_lookup_client: IGeneLookupClient,
        variant_factory: ICreateVariants,
    ) -> None:
        self._entity_annotator = entity_annotator
        self._gene_lookup_client = gene_lookup_client
        self._variant_factory = variant_factory

    def _get_variants(
        self, annotations: Dict[str, Any], query_gene_id: int, query_gene_symbol: str
    ) -> Set[Tuple[str, HGVSVariant]]:
        variants_in_query_gene: Set[Tuple[str, HGVSVariant]] = set()

        if len(annotations) == 0:
            return variants_in_query_gene

        # TODO: filtering based on gene_id is probably too strict - the annotator may have identified a variant but
        # failed to associate it with a gene. This is particularly the case for novel variants if the annotator relies
        # on curated references for variant-gene association.

        for passage in annotations["passages"]:
            for annotation in passage["annotations"]:
                if annotation["infons"]["type"] == "Variant":
                    if "gene_id" in annotation["infons"] and annotation["infons"]["gene_id"] == query_gene_id:
                        if annotation["infons"]["name"] is not None:
                            try:
                                candidate = self._variant_factory.try_parse(
                                    annotation["infons"]["name"], query_gene_symbol
                                )
                                if candidate.valid:
                                    variants_in_query_gene.add((annotation["infons"]["identifier"], candidate))
                            except ValueError:
                                logger.warning(
                                    f"Skipping invalid variant found by PubTator: {annotation['infons']['name']} in "
                                    f"{query_gene_symbol}"
                                )

        return variants_in_query_gene

    def _gather_mentions(self, annotations: Dict[str, Any], variant_id: str) -> Sequence[Dict[str, Any]]:
        mentions: List[Dict[str, Any]] = []

        if len(annotations) == 0:
            return mentions

        for passage in annotations["passages"]:
            for annotation in passage["annotations"]:
                if (
                    annotation["infons"]["type"] == "Variant"
                    and "identifier" in annotation["infons"]
                    and annotation["infons"]["identifier"] == variant_id
                ):
                    to_add = passage.copy()
                    to_add.pop("annotations")
                    mentions.append(to_add)
                    break

        return mentions

    def find_mentions(self, query: str, paper: Paper) -> Dict[HGVSVariant, Sequence[Dict[str, Any]]]:
        """For the VariantMentionFinder, the query is a gene symbol."""
        # Get the gene_id(s) for the query gene(s).
        query_gene_id = self._gene_lookup_client.gene_id_for_symbol(query)

        # Annotate entities in the paper.
        annotations = self._entity_annotator.annotate(paper)

        # Search all of the annotations for `Mutations` with the query gene ids.
        # TODO, loop is no longer necessary if query is only a single gene symbol
        variant_gene_ids: Set[Tuple[str, int, str, HGVSVariant]] = set()
        for gene_symbol, gene_id in query_gene_id.items():
            variants = self._get_variants(annotations, gene_id, gene_symbol)
            variant_gene_tuples = {(gene_symbol, gene_id, variant_id, variant) for variant_id, variant in variants}
            variant_gene_ids.update(variant_gene_tuples)

        # Now collect all the text chunks that mention each variant.
        # Record the gene id for each variant as well.
        mentions: Dict[HGVSVariant, Sequence[Dict[str, Any]]] = {}
        for gene_symbol, gene_id, variant_id, variant in variant_gene_ids:
            mentions[variant] = self._gather_mentions(annotations, variant_id)
            for m in mentions[variant]:
                m["gene_id"] = gene_id
                m["gene_symbol"] = gene_symbol

        return mentions


class TruthsetVariantMentionFinder(IFindVariantMentions):
    def __init__(
        self,
        entity_annotator: IAnnotateEntities,
        gene_lookup_client: IGeneLookupClient,
        variant_factory: ICreateVariants,
    ) -> None:
        self._entity_annotator = entity_annotator
        self._gene_lookup_client = gene_lookup_client
        self._variant_factory = variant_factory

    SHORTHAND_HGVS_PATTERN = re.compile(r"^p\.([A-Za-z]+)(\d+)([A-Za-z\*]|fs|del)$")
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

        hgvsps = self._generate_lowercase_hgvsp_representations(evidence["hgvs_p"]) if evidence["hgvs_p"] else []

        # TODO, this approach to variant nomenclature is pretty brittle. It will only match variant mentions where the
        # hgvs_c or hgvs_p representation in a paper is exactly the same as what's in the truth set (case insensitive).
        # Thus expressions of the variant in paper text that don't conform to hgvs nomenclature standards might be
        # missed.

        # TODO, note that we're simply ignoring gene_id here. This is because pubtator does a relatively poor job of
        # variant-gene linking and results in a lot of false negatives.
        for passage in annotations["passages"]:
            for annotation in passage["annotations"]:
                save = False
                if not annotation["infons"]["type"] == "Variant":
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

    def find_mentions(self, query: str, paper: Paper) -> Dict[HGVSVariant, Sequence[Dict[str, Any]]]:
        """Find variant mentions relevant to `query` that are mentioned in `paper`.

        For TruthsetVariantMentionFinder, the query is a gene symbol.
        """
        truth_rows = [d for d in paper.evidence.values() if d["gene"] == query]

        # TODO, query_gene_ids should really just correspond to a single gene symbol, not a list.
        # Get the gene ids.
        query_gene_ids = self._gene_lookup_client.gene_id_for_symbol(*[v["gene"] for v in truth_rows])

        # Get the annotated paper.
        annotations = self._entity_annotator.annotate(paper)

        # Search all of the annotations for `Mutations` that match the variants from the truth set.
        mentions: Dict[HGVSVariant, Sequence[Dict[str, Any]]] = {}
        for variant_dict in truth_rows:
            gene_symbol = variant_dict["gene"]
            if gene_symbol not in query_gene_ids:
                logger.warning("Gene symbol not found in query gene ids")
                continue

            hgvs_desc = variant_dict["hgvs_c"] if variant_dict["hgvs_c"] else variant_dict["hgvs_p"]
            transcript = variant_dict["transcript"] if variant_dict["transcript"] else None

            variant = self._variant_factory.try_parse(text_desc=hgvs_desc, gene_symbol=gene_symbol, refseq=transcript)
            # It is possible that that there are no mentions for this variant (i.e., this is empty) if the
            # variant is only mentioned in the supplement.
            mentions[variant] = self._gather_mentions_for_variant(
                annotations, query_gene_ids[gene_symbol], variant_dict
            )

        return mentions
