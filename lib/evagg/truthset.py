import csv
import logging
from collections import defaultdict
from functools import cache
from typing import Any, Dict, List, Sequence, Set

from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import ICreateVariants, Paper

from .interfaces import IExtractFields, IGetPapers

logger = logging.getLogger(__name__)


# These are the columns in the truthset that are specific to the paper.
TRUTHSET_PAPER_KEYS = ["paper_id", "pmid", "pmcid", "paper_title", "license", "link"]
TRUTHSET_PAPER_KEYS_MAPPING = {"paper_id": "id", "paper_title": "title"}
# These are the columns in the truthset that are specific to the evidence.
TRUTHSET_EVIDENCE_KEYS = [
    "gene",
    "paper_variant",
    "hgvs_c",
    "hgvs_p",
    "transcript",
    "individual_id",
    "phenotype",
    "zygosity",
    "variant_inheritance",
    "variant_type",
    "study_type",
    "functional_study",  # TODO  Not yet in the truthset.
    "gnomad_frequency",  # TODO  Not yet in the truthset.
]


class TruthsetFileLibrary(IGetPapers, IExtractFields):
    """A class for retrieving papers from a truthset file."""

    _variant_factory: ICreateVariants
    _paper_client: IPaperLookupClient

    def __init__(
        self,
        file_path: str,
        variant_factory: ICreateVariants,
        paper_client: IPaperLookupClient,
        field_map: Sequence[Dict[str, str]],
    ) -> None:
        # Turn list of single-element key-mapping dicts into a list of tuples.
        self._field_map = [kv for [kv] in [kv.items() for kv in field_map]]
        self._file_path = file_path
        self._variant_factory = variant_factory
        self._paper_client = paper_client

    def _process_paper(self, paper_id: str, rows: List[Dict[str, str]]) -> Paper:
        """Process a paper from the truthset file and return a Paper object with associated evidence."""
        logger.info(f"Processing {len(rows)} variants/patients for {paper_id}.")

        # Fetch a Paper object with the extracted fields based on the PMID.
        assert paper_id.startswith("pmid:"), f"Paper ID {paper_id} does not start with 'pmid:'."
        if not (paper := self._paper_client.fetch(paper_id[len("pmid:") :], include_fulltext=True)):
            raise ValueError(f"Failed to fetch paper with ID {paper_id}.")

        for row in rows:
            # Doublecheck the truthset row has the same values as the Paper for all paper-specific keys.
            for row_key in TRUTHSET_PAPER_KEYS:
                key = TRUTHSET_PAPER_KEYS_MAPPING.get(row_key, row_key)
                if paper.props[key] != row[row_key]:
                    raise ValueError(f"Truthset mismatch for {paper.id} {key}: {paper.props[key]} vs {row[row_key]}.")

            # Parse the variant from the HGVS c. or p. description.
            text_desc = row["hgvs_c"] if row["hgvs_c"].startswith("c.") else row["hgvs_p"]
            variant = self._variant_factory.parse(text_desc, row["gene"], row["transcript"])
            assert variant.gene_symbol == row["gene"], f"Gene mismatch {variant}: {variant.gene_symbol}/{row['gene']}"

            # Create an evidence dictionary from the variant/patient-specific columns.
            evidence = {key: row.get(key, "unknown") for key in TRUTHSET_EVIDENCE_KEYS}
            # Add a unique identifier for this combination of paper, variant, and individual ID.
            id = f"{paper.id}_{variant.hgvs_desc}_{row['individual_id']}".replace(" ", "")
            evidence["pub_ev_id"] = id.replace(":", "-").replace("/", "-").replace(">", "-")  # Make it URL-safe.
            # Add this evidence bag to the paper object keyed by the variant and individual ID.
            paper.evidence[(variant, row["individual_id"])] = evidence

        return paper

    @cache
    def _load_truthset(self) -> Set[Paper]:
        row_count = 0
        # Group the rows by paper ID.
        paper_groups = defaultdict(list)
        with open(self._file_path) as tsvfile:
            header = [h.strip() for h in tsvfile.readline().split("\t")]
            if missing_columns := set(TRUTHSET_PAPER_KEYS + TRUTHSET_EVIDENCE_KEYS) - set(header):
                logger.warning(f"Missing columns in truthset table: {missing_columns}")

            reader = csv.reader(tsvfile, delimiter="\t")
            for line in reader:
                fields = dict(zip(header, [field.strip() for field in line]))
                paper_groups[fields["paper_id"]].append(fields)
                row_count += 1

        logger.info(f"Loaded {row_count} rows with {len(paper_groups)} papers from {self._file_path}.")
        # Process each paper row group into a Paper object with truthset evidence filled in.
        papers = {self._process_paper(paper_id, rows) for paper_id, rows in paper_groups.items()}
        # Make sure that each evidence truthset row ID is unique across the truthset.
        assert len({ev["pub_ev_id"] for p in papers for ev in p.evidence.values()}) == row_count
        return papers

    def get_papers(self, query: Dict[str, Any]) -> Sequence[Paper]:
        """For the TruthsetFileLibrary, query is expected to be a gene symbol."""
        all_papers = self._load_truthset()

        if gene_symbol := query.get("gene_symbol"):
            # Filter to just the papers with variants that have evidence for the gene specified in the query.
            return [p for p in all_papers if gene_symbol in {v[0].gene_symbol for v in p.evidence.keys()}]
        return []

    # IExtractFields
    def extract(self, paper: Paper, gene_symbol: str) -> Sequence[Dict[str, str]]:
        """Extract properties from the evidence bags populated on the truthset Paper object."""

        def _get_props(evidence: Dict[str, str]) -> Dict[str, str]:
            """Extract the requested evidence properties from the paper and truthset evidence bag."""
            return {out_key: (paper.props | evidence)[k] for k, out_key in self._field_map}

        # For each evidence set in the paper that has a matching gene, extract the evidence properties.
        extracted_fields = [_get_props(ev) for ev in paper.evidence.values() if ev["gene"] == gene_symbol]
        return extracted_fields
