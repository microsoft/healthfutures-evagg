import logging
import re
from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

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
            return self._refseq_client.transcript_accession_for_symbol(gene_symbol)
        elif text_desc.startswith("m."):
            return self.MITO_REFSEQ
        elif text_desc.startswith("g."):
            raise ValueError(f"Genomic (g. prefixed) variants must have a RefSeq. None was provided for {text_desc}")
        else:
            logger.warning(
                "Unable to predict refseq for variant with unknown HGVS "
                f"type: {text_desc} with gene symbol {gene_symbol}."
            )
            return None

    def _clean_refseq(self, refseq: str | None, text_desc: str, gene_symbol: str | None) -> Tuple[str, bool]:
        # If no refseq is provided, we'll try to predict it. If one is provided, we'll make sure it's versioned
        # correctly and complete.
        if not refseq:
            refseq = self._predict_refseq(text_desc, gene_symbol)
            refseq_predicted = True
        elif refseq.find(".") < 0:
            refseq_replacement = self._refseq_client.accession_autocomplete(refseq)
            if refseq_replacement:
                refseq = refseq_replacement
                refseq_predicted = True
            else:
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

        return refseq, refseq_predicted

    def parse_rsid(self, rsid: str) -> HGVSVariant:
        """Parse a variant based on an rsid."""
        hgvs_lookup = self._variant_lookup_client.hgvs_from_rsid(rsid)
        full_hgvs = None
        gene_symbol = None

        if rsid in hgvs_lookup:
            if "hgvs_c" in hgvs_lookup[rsid]:
                full_hgvs = hgvs_lookup[rsid]["hgvs_c"]
            elif "hgvs_p" in hgvs_lookup[rsid]:
                full_hgvs = hgvs_lookup[rsid]["hgvs_p"]
            elif "hgvs_g" in hgvs_lookup[rsid]:
                full_hgvs = hgvs_lookup[rsid]["hgvs_g"]
            gene_symbol = hgvs_lookup[rsid].get("gene")

        if not full_hgvs:
            raise ValueError(f"Could not find HGVS for info rsid {rsid}")

        refseq = full_hgvs.split(":")[0]
        text_desc = full_hgvs.split(":")[1]

        return self.parse(text_desc, gene_symbol, refseq)

    def _normalize_and_create(
        self, text_desc: str, gene_symbol: str | None, refseq: str, refseq_predicted: bool
    ) -> HGVSVariant:
        normalized = self._normalizer.normalize(f"{refseq}:{text_desc}")

        # Normalize the variant description.
        if "normalized_description" in normalized:
            normalized_hgvs = normalized["normalized_description"]
            refseq = normalized_hgvs.split(":")[0]
            new_text_desc = normalized_hgvs.split(":")[1]
            if new_text_desc.find("(") >= 0 and new_text_desc.find(")") >= 0:
                text_desc = new_text_desc.replace("(", "").replace(")", "")
            else:
                text_desc = new_text_desc
            # If this is a protein variant, we only want the NP_ portion of the refseq.
            if text_desc.startswith("p."):
                match = re.search(r"NP_\d+\.\d+", refseq)
                refseq = match.group(0) if match else refseq

        # If this is a genomic variant and there are coding equivalents, make them.
        coding_equivalents: List[HGVSVariant] = []
        if (
            text_desc.startswith("g.")
            and "equivalent_descriptions" in normalized
            and "c" in normalized["equivalent_descriptions"]
        ):
            for coding_description in normalized["equivalent_descriptions"]["c"]:
                coding_refseq, coding_text_desc = coding_description["description"].split(":")
                coding_equivalents.append(
                    self._normalize_and_create(coding_text_desc, gene_symbol, coding_refseq, refseq_predicted)
                )

        # If there's a protein consequence, make it
        if "protein" in normalized:
            protein_refseq, protein_text_desc = normalized["protein"]["description"].split(":")
            protein_consequence = self._normalize_and_create(
                protein_text_desc, gene_symbol, protein_refseq, refseq_predicted
            )
        else:
            protein_consequence = None

        # Construct and return the resulting variant.
        return HGVSVariant(
            hgvs_desc=text_desc,
            gene_symbol=gene_symbol,
            refseq=refseq,
            refseq_predicted=refseq_predicted,
            valid=normalized.get("error_message", None) is None,
            validation_error=normalized.get("error_message", None),
            protein_consequence=protein_consequence,
            coding_equivalents=coding_equivalents,
        )

    def parse(self, text_desc: str, gene_symbol: str | None, refseq: str | None = None) -> HGVSVariant:
        """Attempt to parse a variant based on description and an optional gene symbol and optional refseq.

        `gene_symbol` is required for protein (p.) and coding (c.) variants, but not mitochondrial (m.) or genomic (g.)
        variants.

        `refseq` is required for genomic (g.) variants. For other variant types, if a `refseq` is not provided, it will
        be predicted based on the variant description gene symbol.

        Raises a ValueError if the above requirements are not met. Otherwise, this function will attempt to parse the
        provided variant description and return a validated, normalized representation of that variant. If this is not
        possible, an invalid, non-normalized representation of the variant description will be returned.
        """
        if (text_desc.startswith("p.") or text_desc.startswith("c.")) and not gene_symbol:
            raise ValueError(f"Gene symbol required for protein and coding variants: {text_desc}")

        refseq, refseq_predicted = self._clean_refseq(refseq, text_desc, gene_symbol)
        (is_valid, validation_error) = self._validator.validate(f"{refseq}:{text_desc}")

        # From here, if the variant is valid (or if it's a frameshift) we make a normalized variant (recursing as
        # necessary). Otherwise we make a non-normalized variant.
        if (
            is_valid or text_desc.find("fs") >= 0
        ):  # frame shift variants are not validated by mutalyzer, but they can be normalized.
            return self._normalize_and_create(text_desc, gene_symbol, refseq, refseq_predicted)
        else:
            return HGVSVariant(
                hgvs_desc=text_desc,
                gene_symbol=gene_symbol,
                refseq=refseq,
                refseq_predicted=refseq_predicted,
                valid=False,
                validation_error=validation_error,
                protein_consequence=None,
                coding_equivalents=[],
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
        return {
            tok.rstrip(")").split(".")[0]: (int(tok.rstrip(")").split(".")[1]) if "." in tok else -1)
            for tok in refseq.split("(")
        }

    def _more_complete_by_refseq(
        self, variant1: HGVSVariant, variant2: HGVSVariant, allow_mismatch: bool = False
    ) -> HGVSVariant:
        v1score = self._score_refseq_completeness(variant1)
        v2score = self._score_refseq_completeness(variant2)

        # If neither variant has a refseq, we would have already determined them as equivalent.

        if v1score > v2score:
            return variant1
        elif v1score < v2score:
            return variant2

        # All other things being equal, if they have the same refseq accession, keep the one with the highest version.
        v1rs_dict = self._parse_refseq_parts(variant1.refseq) if variant1.refseq else {}
        v2rs_dict = self._parse_refseq_parts(variant2.refseq) if variant2.refseq else {}

        shared_keys = v1rs_dict.keys() & v2rs_dict.keys()
        assert not (shared_keys and allow_mismatch)

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
        - if the coding equivalent of one is the same variant as the other
        - if the protein consequence of the coding equivalent of one is the same variant as the other
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

        if variant2.hgvs_desc.startswith("g.") and any(
            self._fuzzy_compare(variant1, v, disregard_refseq)
            or (v.protein_consequence and self._fuzzy_compare(variant1, v.protein_consequence, disregard_refseq))
            for v in variant2.coding_equivalents
        ):
            # Variant1 is the coding or protein variant, either of which is more common than the genomic variant.
            return variant1

        if variant1.hgvs_desc.startswith("g.") and any(
            self._fuzzy_compare(variant2, v, disregard_refseq)
            or (v.protein_consequence and self._fuzzy_compare(variant2, v.protein_consequence, disregard_refseq))
            for v in variant1.coding_equivalents
        ):
            # Variant2 is the coding or protein variant, either of which is more common than the genomic variant.
            return variant2

        return None
