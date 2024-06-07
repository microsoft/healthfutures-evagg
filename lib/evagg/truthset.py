import csv
import logging
from functools import cache
from typing import Any, Dict, List, Optional, Sequence

from lib.evagg.content.fulltext import get_sections
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import HGVSVariant, ICreateVariants, Paper

from .content import IFindObservations, Observation
from .interfaces import IExtractFields, IGetPapers

logger = logging.getLogger(__name__)


# These are the columns in the truthset that are specific to the paper.
TRUTHSET_PAPER_KEYS = ["paper_id", "pmid", "pmcid", "paper_title", "license", "link"]
TRUTHSET_PAPER_KEYS_MAPPING = {"paper_id": "id", "paper_title": "title"}


class TruthsetFileHandler(IGetPapers, IFindObservations, IExtractFields):
    """A class for retrieving papers from a truthset file."""

    def __init__(
        self,
        file_path: str,
        variant_factory: ICreateVariants,
        paper_client: IPaperLookupClient,
        fields: Optional[Sequence[str]] = None,
    ) -> None:
        self._file_path = file_path
        self._variant_factory = variant_factory
        self._paper_client = paper_client
        self._fields = fields or []

    @cache
    def _get_evidence(self) -> Sequence[Dict[str, Any]]:
        """Load the truthset evidence from the file and return it as a list of dictionaries."""
        with open(self._file_path) as tsvfile:
            column_names = [c.strip() for c in tsvfile.readline().split("\t")]
            evidence = [dict(zip(column_names, row)) for row in csv.reader(tsvfile, delimiter="\t")]

        paper_count = len({ev["paper_id"] for ev in evidence})
        logger.info(f"Loaded {len(evidence)} rows with {paper_count} papers from {self._file_path}.")
        return evidence

    def _parse_variant(self, ev: Dict[str, str]) -> HGVSVariant:
        """Parse the variant from the HGVS c. or p. description."""
        text_desc = ev["hgvs_c"] if ev["hgvs_c"].startswith("c.") else ev["hgvs_p"]
        variant = self._variant_factory.parse(text_desc, ev["gene"], ev["transcript"])
        assert variant.gene_symbol == ev["gene"], f"Gene mismatch {variant}: {variant.gene_symbol}/{ev['gene']}"
        return variant

    # IGetPapers
    def get_papers(self, query: Dict[str, Any]) -> Sequence[Paper]:
        """For the TruthsetFileHandler, query is expected to be a gene symbol."""
        if not (gene_symbol := query.get("gene_symbol")):
            logger.warning("No gene symbol provided for truthset query.")
            return []

        papers: List[Paper] = []
        # Loop over all paper ids for evidence rows that match the gene symbol.
        for paper_id in {ev["paper_id"] for ev in self._get_evidence() if ev["gene"] == gene_symbol}:
            # Fetch a Paper object with the extracted fields based on the PMID.
            assert paper_id.startswith("pmid:"), f"Paper ID {paper_id} does not start with 'pmid:'."
            if not (paper := self._paper_client.fetch(paper_id[len("pmid:") :], include_fulltext=True)):
                raise ValueError(f"Failed to fetch paper with ID {paper_id}.")
            # Validate the truthset rows have the same values as the Paper for all paper-specific keys.
            for row in [ev for ev in self._get_evidence() if ev["paper_id"] == paper_id]:
                for row_key in TRUTHSET_PAPER_KEYS:
                    k = TRUTHSET_PAPER_KEYS_MAPPING.get(row_key, row_key)
                    if paper.props[k] != row[row_key]:
                        raise ValueError(f"Truthset mismatch for {paper.id} {k}: {paper.props[k]} vs {row[row_key]}.")
            # Add the paper to the list of papers.
            papers.append(paper)

        return papers

    async def find_observations(self, gene_symbol: str, paper: Paper) -> Sequence[Observation]:
        """Identify all observations relevant to `gene_symbol` in `paper`."""
        if not (paper.props.get("fulltext_xml")):
            logger.warning(f"Skipping {paper.id} because full text could not be retrieved")
            return []

        def _get_observation(evidence: Dict[str, str]) -> Observation:
            """Create an Observation object from the evidence dictionary."""
            individual = evidence["individual_id"]
            # TODO, consider filtering to relevant sections.
            texts = list(get_sections(paper.props["fulltext_xml"]))
            # Parse the variant from the evidence values.
            variant = self._parse_variant(evidence)
            # Accumulate the various descriptions for the variant.
            variant_descriptions = {variant.hgvs_desc, evidence["paper_variant"]}
            if variant.protein_consequence:
                variant_descriptions |= {variant.protein_consequence.hgvs_desc}
            return Observation(variant, individual, list(variant_descriptions), [individual], texts)

        observations = [
            _get_observation(evidence)
            for evidence in self._get_evidence()
            if evidence["paper_id"] == paper.id and evidence["gene"] == gene_symbol
        ]
        return observations

    # IExtractFields
    def extract(self, paper: Paper, gene_symbol: str) -> Sequence[Dict[str, str]]:
        """Extract properties from the evidence bags populated on the truthset Paper object."""
        if not self._fields:
            raise ValueError("TruthsetFileHandler not configured for field extraction.")

        def _get_field(evidence: Dict[str, str], field: str) -> str:
            """Extract the requested evidence properties from the truthset evidence."""
            if field == "pub_ev_id":
                # Create a unique identifier for this combination of paper, variant, and individual ID.
                value = self._parse_variant(evidence).get_unique_id(evidence["paper_id"], evidence["individual_id"])
            elif field in evidence:
                value = evidence[field]
            elif field in paper.props:
                value = paper.props[field]
            elif field == "gnomad_frequency":
                value = "TODO"  # TODO  Not yet in the truthset.
            else:
                raise ValueError(f"Unsupported field: {field}")
            return value

        extracted_fields = [
            {field: _get_field(evidence, field) for field in self._fields}
            for evidence in self._get_evidence()
            if evidence["paper_id"] == paper.id and evidence["gene"] == gene_symbol
        ]
        return extracted_fields
