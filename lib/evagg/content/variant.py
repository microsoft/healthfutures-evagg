import logging
from collections import defaultdict
from typing import Dict, Sequence, Set

from lib.evagg.ref import INormalizeVariants, IRefSeqLookupClient, IValidateVariants, IVariantLookupClient
from lib.evagg.types import HGVSVariant, ICreateVariants

from .interfaces import ICompareVariants

logger = logging.getLogger(__name__)


class HGVSVariantFactory(ICreateVariants):
    _validator: IValidateVariants
    _normalizer: INormalizeVariants
    _variant_lookup_client: IVariantLookupClient
    _refseq_client: IRefSeqLookupClient

    MITO_REFSEQ = "NC_012920.1"

    def __init__(
        self,
        validator: IValidateVariants,
        normalizer: INormalizeVariants,
        variant_lookup_client: IVariantLookupClient,
        refseq_client: IRefSeqLookupClient,
    ) -> None:
        self._validator = validator
        self._normalizer = normalizer
        self._variant_lookup_client = variant_lookup_client
        self._refseq_client = refseq_client

    def _predict_refseq(self, text_desc: str, gene_symbol: str | None) -> str | None:
        """Predict the RefSeq for a variant based on its description and gene symbol."""
        if text_desc.startswith("p.") and gene_symbol:
            protein_accession = self._refseq_client.protein_accession_for_symbol(gene_symbol)
            if transcript_accession := self._refseq_client.transcript_accession_for_symbol(gene_symbol):
                return f"{transcript_accession}({protein_accession})"
            return protein_accession
        elif text_desc.startswith("c.") and gene_symbol:
            # TODO: consider also pulling the genomic refseq for the gene_symbol?
            return self._refseq_client.transcript_accession_for_symbol(gene_symbol)
        elif text_desc.startswith("m."):
            # TODO: consider moving?
            return self.MITO_REFSEQ
        elif text_desc.startswith("g."):
            raise ValueError(f"Genomic (g. prefixed) variants must have a RefSeq. None was provided for {text_desc}")
        else:
            logger.warning(f"Unsupported HGVS type: {text_desc} with gene symbol {gene_symbol}")
            return None

    def parse_rsid(self, rsid: str) -> HGVSVariant:
        """Parse a variant based on an rsid."""
        hgvs_lookup = self._variant_lookup_client.hgvs_from_rsid(rsid)
        full_hgvs = None
        if rsid in hgvs_lookup:
            if "hgvs_c" in hgvs_lookup[rsid]:
                full_hgvs = hgvs_lookup[rsid]["hgvs_c"]
            elif "hgvs_p" in hgvs_lookup[rsid]:
                full_hgvs = hgvs_lookup[rsid]["hgvs_p"]

        if not full_hgvs:
            raise ValueError(f"Could not find HGVS for info rsid {rsid}")

        refseq = full_hgvs.split(":")[0]
        text_desc = full_hgvs.split(":")[1]

        return self.parse(text_desc, None, refseq)

    def parse(self, text_desc: str, gene_symbol: str | None, refseq: str | None = None) -> HGVSVariant:
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
        # TODO: if we're going to infer the chromosomal description from hgvs description and gene symbol, we need
        # to know the genome build.

        # If no refseq is provided, we'll try to predict it. If one is provided, we'll make sure it's versioned
        # correctly.
        if not refseq:
            refseq = self._predict_refseq(text_desc, gene_symbol)
            refseq_predicted = True
        elif refseq.find(".") < 0:
            refseq_replacement = self._refseq_client.accession_autocomplete(refseq)
            if refseq_replacement:
                refseq = refseq_replacement
                refseq_predicted = True
            refseq_predicted = False
        else:
            refseq_predicted = False

        if not refseq:
            raise ValueError(f"No RefSeq provided or predicted for {text_desc, gene_symbol}")

        # If the variant is intronic, the refseq should be either a transcript with an included genomic reference, or
        # should be a standalone genomic reference.
        #   see https://hgvs-nomenclature.org/stable/background/refseq/
        if text_desc.find("+") >= 0 or text_desc.find("-") >= 0:
            # Intronic sequence variant, ensure that we've got an NG_ or NC_ refseq to start
            if refseq.startswith("NM_"):
                # Find the associated genomic reference sequence.
                logger.debug(
                    f"Intronic variant without genomic reference, attempting to fix: {text_desc} {gene_symbol}"
                )
                if gene_symbol:
                    chrom_refseq = self._refseq_client.genomic_accession_for_symbol(gene_symbol)
                    refseq = f"{chrom_refseq}({refseq})" if chrom_refseq else refseq

        is_valid = self._validator.validate(f"{refseq}:{text_desc}")

        protein_consequence = None
        if (
            is_valid or text_desc.find("fs") >= 0
        ):  # frame shift variants are not validated by mutalyzer, but they can be normalized.
            normalized = self._normalizer.normalize(f"{refseq}:{text_desc}")

            # Normalize the variant.
            if "normalized_description" in normalized:
                normalized_hgvs = normalized["normalized_description"]
                refseq = normalized_hgvs.split(":")[0]
                new_text_desc = normalized_hgvs.split(":")[1]
                if new_text_desc.find("(") >= 0 and new_text_desc.find(")") >= 0:
                    text_desc = new_text_desc.replace("(", "").replace(")", "")
                else:
                    text_desc = new_text_desc

            # If there's a protein consequence, keep it handy.
            if "protein" in normalized:
                protein_hgvs = normalized["protein"].get("description", "")
                # protein description should be NM_1234.1(NP_1234.1):p.(Arg123Gly) or NG_ in place of NM, extract the
                # NP_ and p. parts.
                if protein_hgvs.find(":") >= 0:
                    protein_desc = protein_hgvs.split(":")[1].replace("(", "").replace(")", "")
                    protein_refseq = protein_hgvs.split(":")[0].split("(")[1].split(")")[0]

                    protein_consequence = HGVSVariant(
                        hgvs_desc=protein_desc,
                        gene_symbol=gene_symbol,
                        refseq=protein_refseq,
                        refseq_predicted=True,
                        valid=True,
                        protein_consequence=None,
                    )

        return HGVSVariant(
            hgvs_desc=text_desc,
            gene_symbol=gene_symbol,
            refseq=refseq,
            refseq_predicted=refseq_predicted,
            valid=is_valid,
            protein_consequence=protein_consequence,
        )


class HGVSVariantComparator(ICompareVariants):

    def _score_refseq_completeness(self, v: HGVSVariant) -> int:
        """Return a score for the completeness refseq for a variant."""
        # Scores are assigned as follows:
        #   3) having a refseq that is not predicted
        #   2) having both a refseq and a selector
        #   1) having a refseq at all

        if not v.refseq:
            return 0
        if not v.refseq_predicted:
            return 3
        elif v.refseq.find("(") >= 0:
            return 2
        # At least we have a refseq.
        return 1

    def _parse_refseq_parts(self, refseq: str) -> Dict[str, int]:
        """Parse a refseq accession string into a dictionary of accessions and versions."""
        return {tok.rstrip(")").split(".")[0]: int(tok.rstrip(")").split(".")[1]) for tok in refseq.split("(")}

    def _more_complete_by_refseq(
        self, variant1: HGVSVariant, variant2: HGVSVariant, allow_mismatch: bool = False
    ) -> HGVSVariant:
        v1score = self._score_refseq_completeness(variant1)
        v2score = self._score_refseq_completeness(variant2)

        if v1score == 0 and v2score == 0:
            # Neither has a refseq.
            return variant1

        if v1score > v2score:
            return variant1
        elif v1score < v2score:
            return variant2

        # All other things being equal, if they have the same refseq accession, keep the one with the highest version.
        v1rs_dict = self._parse_refseq_parts(variant1.refseq) if variant1.refseq else {}
        v2rs_dict = self._parse_refseq_parts(variant2.refseq) if variant2.refseq else {}

        shared_keys = v1rs_dict.keys() & v2rs_dict.keys()

        if not shared_keys and not allow_mismatch:
            raise ValueError(
                "Expectation mismatch: comparing refseq completeness for variants with no shared accessions."
            )

        for k in shared_keys:
            if v1rs_dict[k] > v2rs_dict[k]:
                return variant1
            elif v1rs_dict[k] < v2rs_dict[k]:
                return variant2

        # If none of the above conditions are met, keep the first one but log.
        logger.info(f"Unable to determine more complete variant based on refseq completeness: {variant1}, {variant2}")
        return variant1

    def consolidate(
        self, variants: Sequence[HGVSVariant], disregard_refseq: bool = False
    ) -> Dict[HGVSVariant, Set[HGVSVariant]]:
        """Consolidate equivalent variants.

        Return a mapping from the retained variants to all variants collapsed into that variant.
        """
        consolidated_variants: Dict[HGVSVariant, Set[HGVSVariant]] = defaultdict(set)

        for variant in variants:
            found = False

            for saved_variant in consolidated_variants.keys():
                # - if they are the same variant
                if (keeper := self.compare(variant, saved_variant, disregard_refseq)) is not None:
                    if saved_variant == keeper:
                        consolidated_variants[saved_variant].add(variant)
                    else:  # variant == keeper
                        matches = consolidated_variants.pop(saved_variant)
                        matches.add(variant)
                        consolidated_variants[variant] = matches
                    found = True
                    break

            # It's a new variant, save it.
            if not found:
                consolidated_variants[variant] = {variant}

        return consolidated_variants

    def _fuzzy_compare(self, variant1: HGVSVariant, variant2: HGVSVariant, disregard_refseq: bool) -> bool:
        """Compare a variant to a protein consequence."""
        desc_match = variant1.hgvs_desc.replace("(", "").replace(")", "") == variant2.hgvs_desc.replace(
            "(", ""
        ).replace(")", "")

        if not desc_match:
            return False
        else:
            if disregard_refseq:
                return True
            else:
                # We know at least one of the refseqs is not None, otherwise the variants would already have been
                # determined to be equivalent.
                v1accessions = (
                    {tok.rstrip(")").split(".")[0] for tok in variant1.refseq.split("(")} if variant1.refseq else set()
                )
                v2accessions = (
                    {tok.rstrip(")").split(".")[0] for tok in variant2.refseq.split("(")} if variant2.refseq else set()
                )

                return bool(v1accessions.intersection(v2accessions))

    def compare(
        self, variant1: HGVSVariant, variant2: HGVSVariant, disregard_refseq: bool = False
    ) -> HGVSVariant | None:
        """Compare to variants for biological equivalence.

        The logic below is a little gnarly, so it's worth a short description here. Effectively this method is
        attempting to remove any redundancy in `variants` so any keys in that dict that are biologically "linked" are
        merged, and the string descriptions of those variants that were observed in the paper are retained.

        Note that variants are be biologically linked under the following conditions:
        - if they are the same variant
        - if the protein consequence of one is the the same variant as the other
        - they have the same hgvs_description and share at least one refseq accession (regardless of version)

        Optionally, one can disregard the refseq entirely.

        Note: this method draws no distinction between valid and invalid variants, invalid variants will not be
        normalized, nor will they have a protein consequence, so they are very likely to be considered distinct from
        other biologically linked variants.

        Return the more complete variant if they are equivalent, None if they are not.
        """
        if variant1 == variant2:
            # Direct equivalence of the variant class handles the easy stuff.
            return variant1

        if self._fuzzy_compare(variant1, variant2, disregard_refseq):
            # Same desc, determine the more complete variant WRT refseq.
            # If disregard_refseq == False, these two variants are guaranteed to share one or more refseq accessions.
            return self._more_complete_by_refseq(variant1, variant2, allow_mismatch=disregard_refseq)

        if variant2.protein_consequence and self._fuzzy_compare(
            variant1, variant2.protein_consequence, disregard_refseq
        ):
            # Variant2 is DNA, so more complete.
            return variant2

        if variant1.protein_consequence and self._fuzzy_compare(
            variant1.protein_consequence, variant2, disregard_refseq
        ):
            # Variant1 is DNA, so more complete.
            return variant1

        return None
