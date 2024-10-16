import csv
import logging
from functools import cache
from typing import Any, Dict, List, Optional, Sequence

from lib.evagg.content.fulltext import get_sections
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import HGVSVariant, ICreateVariants, Paper

from .content import IFindObservations, Observation
from .interfaces import IGetPapers
from .simple import PropertyContentExtractor

logger = logging.getLogger(__name__)


# These are the columns in the truthset that are specific to the paper.
TRUTHSET_PAPER_KEYS = ["paper_id", "pmid", "pmcid", "paper_title", "license", "link"]
TRUTHSET_PAPER_KEYS_MAPPING = {"paper_id": "id", "paper_title": "title"}


class TruthsetFileHandler(IGetPapers, IFindObservations, PropertyContentExtractor):
    """A class for retrieving papers from a truthset file."""

    def __init__(
        self,
        file_path: str,
        variant_factory: ICreateVariants,
        paper_client: IPaperLookupClient,
        fields: Optional[Sequence[str]] = None,
    ) -> None:
        PropertyContentExtractor.__init__(self, fields or [])
        self._file_path = file_path
        self._variant_factory = variant_factory
        self._paper_client = paper_client

    @cache
    def _get_all_evidence(self) -> Sequence[Dict[str, Any]]:
        """Load the truthset evidence from the file and return it as a list of dictionaries."""
        with open(self._file_path) as tsvfile:
            column_names = [c.strip() for c in tsvfile.readline().split("\t")]
            evidence = [dict(zip(column_names, row)) for row in csv.reader(tsvfile, delimiter="\t")]

        paper_count = len({ev["paper_id"] for ev in evidence})
        logger.info(f"Loaded {len(evidence)} rows with {paper_count} papers from {self._file_path}.")
        return evidence

    def _get_evidence(
        self, paper_id: Optional[str] = None, gene_symbol: Optional[str] = None
    ) -> Sequence[Dict[str, Any]]:
        """Return the evidence rows that match the paper_id and gene_symbol."""
        return [
            evidence
            for evidence in self._get_all_evidence()
            if (paper_id is None or evidence["paper_id"] == paper_id)
            and (gene_symbol is None or evidence["gene"] == gene_symbol)
        ]

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
        # Loop over all paper ids for evidence rows that match the gene symbol in order of paper_id.
        for paper_id in sorted({ev["paper_id"] for ev in self._get_evidence(gene_symbol=gene_symbol)}):
            # Fetch a Paper object with the extracted fields based on the PMID.
            assert paper_id.startswith("pmid:"), f"Paper ID {paper_id} does not start with 'pmid:'."
            if not (paper := self._paper_client.fetch(paper_id[len("pmid:") :], include_fulltext=True)):
                raise ValueError(f"Failed to fetch paper with ID {paper_id}.")
            # Validate the truthset rows have the same values
            # as the Paper for all paper-specific keys.
            for row in self._get_evidence(paper_id=paper_id):
                for row_key in TRUTHSET_PAPER_KEYS:
                    k = TRUTHSET_PAPER_KEYS_MAPPING.get(row_key, row_key)
                    if paper.props[k] != row[row_key]:
                        raise ValueError(f"Truthset mismatch for {paper.id} {k}: {paper.props[k]} vs {row[row_key]}.")
            # Add the paper to the list of papers.
            papers.append(paper)

        return papers

    async def find_observations(self, gene_symbol: str, paper: Paper) -> Sequence[Observation]:
        """Identify all observations relevant to `gene_symbol` in `paper`."""
        if not (paper.props.get("can_access")):
            logger.warning(f"Skipping {paper.id} because full text could not be retrieved")
            return []

        def _get_observation(evidence: Dict[str, str]) -> Observation:
            """Create an Observation object from the evidence dictionary."""
            individual = evidence["individual_id"]
            texts = list(get_sections(paper.props["fulltext_xml"]))
            # Parse the variant from the evidence values.
            variant = self._parse_variant(evidence)
            # Accumulate the various descriptions for the variant.
            variant_descriptions = {variant.hgvs_desc, evidence["paper_variant"]}
            if variant.protein_consequence:
                variant_descriptions |= {variant.protein_consequence.hgvs_desc}
            return Observation(variant, individual, list(variant_descriptions), [individual], texts, paper.id)

        return [_get_observation(evidence) for evidence in self._get_evidence(paper.id, gene_symbol)]

    # PropertyContentExtractor/IExtractFields
    def get_evidence(self, paper: Paper, gene_symbol: str) -> Sequence[Dict[str, str]]:
        def _add_fields(ev: Dict[str, str]) -> Dict[str, str]:
            """Add a unique identifier for the evidence."""
            if "evidence_id" in self._fields:
                ev["evidence_id"] = self._parse_variant(ev).get_unique_id(ev["paper_id"], ev["individual_id"])
            if "citation" in self._fields:
                ev["citation"] = paper.props["citation"]
            if "link" in self._fields:
                ev["link"] = paper.props["link"]
            if "gnomad_frequency" in self._fields:
                ev["gnomad_frequency"] = "unknown"
            return ev

        return [_add_fields(ev) for ev in self._get_evidence(paper.id, gene_symbol)]
