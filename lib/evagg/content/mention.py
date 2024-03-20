import logging

from lib.evagg.ref import INormalizeVariants, IRefSeqLookupClient, IValidateVariants
from lib.evagg.types import HGVSVariant, ICreateVariants

logger = logging.getLogger(__name__)


class HGVSVariantFactory(ICreateVariants):
    _validator: IValidateVariants
    _normalizer: INormalizeVariants
    _refseq_client: IRefSeqLookupClient

    MITO_REFSEQ = "NC_012920.1"

    def __init__(
        self,
        validator: IValidateVariants,
        normalizer: INormalizeVariants,
        refseq_client: IRefSeqLookupClient,
    ) -> None:
        self._validator = validator
        self._normalizer = normalizer
        self._refseq_client = refseq_client

    def _predict_refseq(self, text_desc: str, gene_symbol: str | None) -> str | None:
        """Predict the RefSeq for a variant based on its description and gene symbol."""
        if text_desc.startswith("p.") and gene_symbol:
            return self._refseq_client.protein_accession_for_symbol(gene_symbol)
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
                logger.info(f"Intronic variant without genomic reference, attempting to fix: {text_desc} {gene_symbol}")
                if gene_symbol:
                    chrom_refseq = self._refseq_client.genomic_accession_for_symbol(gene_symbol)
                    refseq = f"{chrom_refseq}({refseq})" if chrom_refseq else refseq

        is_valid = self._validator.validate(f"{refseq}:{text_desc}")

        protein_consequence = None
        if is_valid:
            normalized = self._normalizer.normalize(f"{refseq}:{text_desc}")

            # Normalize the variant.
            if "normalized_description" in normalized:
                normalized_hgvs = normalized["normalized_description"]
                refseq = normalized_hgvs.split(":")[0]
                new_text_desc = normalized_hgvs.split(":")[1]
                if text_desc.find("(") < 0:
                    text_desc = new_text_desc.replace("(", "").replace(")", "")
                else:
                    text_desc = new_text_desc

            # If there's a protein consequence, keep it handy.
            if "protein" in normalized:
                protein_hgvs = normalized["protein"].get("description", "")
                # protein description should be NM_1234.1(NP_1234.1):p.(Arg123Gly) or NG_ in place of NM, extract the
                # NP_ and p. parts.
                if protein_hgvs.find(":") >= 0:
                    protein_desc = protein_hgvs.split(":")[1]
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
