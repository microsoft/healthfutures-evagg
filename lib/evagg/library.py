import csv
import json
import logging
import os
from collections import defaultdict
from functools import cache
from typing import Dict, Sequence, Set, Tuple

from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import HGVSVariant, ICreateVariants, Paper

from .interfaces import IGetPapers

logger = logging.getLogger(__name__)


class SimpleFileLibrary(IGetPapers):
    def __init__(self, collections: Sequence[str]) -> None:
        self._collections = collections

    def _load_collection(self, collection: str) -> Dict[str, Paper]:
        papers = {}
        # collection is a local directory, get a list of all of the json files in that directory
        for filename in os.listdir(collection):
            if filename.endswith(".json"):
                # load the json file into a dict and append it to papers
                with open(os.path.join(collection, filename), "r") as f:
                    paper = Paper(**json.load(f))
                    papers[paper.id] = paper
        return papers

    @cache
    def _load(self) -> Dict[str, Paper]:
        papers = {}
        for collection in self._collections:
            papers.update(self._load_collection(collection))

        return papers

    def search(self, query: str) -> Set[Paper]:
        # Dummy implementation that returns all papers regardless of query.
        all_papers = set(self._load().values())
        return all_papers


# These are the columns in the truthset that are specific to the paper.
TRUTHSET_PAPER_KEYS = ["paper_id", "pmid", "pmcid", "paper_title", "link", "is_pmc_oa", "license"]
# These are the columns in the truthset that are specific to the variant.
TRUTHSET_VARIANT_KEYS = [
    "gene",
    "transcript",
    "hgvs_c",
    "hgvs_p",
    "individual_id",
    "phenotype",
    "zygosity",
    "variant_inheritance",
    "study_type",
    "functional_study",
    "variant_type",
    "notes",
]


class TruthsetFileLibrary(IGetPapers):
    """A class for retrieving papers from a truthset file."""

    _variant_factory: ICreateVariants

    def __init__(self, file_path: str, variant_factory: ICreateVariants) -> None:
        self._file_path = file_path
        self._variant_factory = variant_factory

    @cache
    def _load_truthset(self) -> Set[Paper]:
        # Group the rows by paper ID.
        paper_groups = defaultdict(list)
        with open(self._file_path) as tsvfile:
            header = [h.strip() for h in tsvfile.readline().split("\t")]
            reader = csv.reader(tsvfile, delimiter="\t")
            for line in reader:
                fields = dict(zip(header, [field.strip() for field in line]))
                paper_id = fields.get("paper_id")
                paper_groups[paper_id].append(fields)

        papers: Set[Paper] = set()
        for paper_id, rows in paper_groups.items():
            if paper_id == "MISSING_ID":
                logger.warning(f"Skipped {len(rows)} rows with no paper ID.")
                continue

            # For each paper, extract the paper-specific key/value pairs into a new dict.
            # These are repeated on every paper/variant row, so we can just take the first row.
            paper_data = {k: v for k, v in rows[0].items() if k in TRUTHSET_PAPER_KEYS}

            # Integrity checks.
            for row in rows:
                # Doublecheck if every row has the same values for the paper-specific keys.
                for key in TRUTHSET_PAPER_KEYS:
                    if paper_data[key] != row[key]:
                        logger.warning(f"Multiple values ({paper_data[key]} vs {row[key]}) for {key} ({paper_id}).")
                # Make sure the gene/variant columns are not empty.
                if not row["gene"] or (not row["hgvs_p"] and not row["hgvs_c"]):
                    logger.warning(f"Missing gene or hgvs_p for {paper_id}.")

            # Return the parsed variant from HGVS c or p and the individual ID.
            def _get_variant_key(row: Dict[str, str]) -> Tuple[HGVSVariant | None, str]:
                text_desc = row["hgvs_c"] if row["hgvs_c"].startswith("c.") else row["hgvs_p"]
                transcript = row["transcript"] if "transcript" in row else None

                # This can fail, instead of raising an exception, we'll return a placeholder value that can be dropped
                # from the set of variants later.
                try:
                    return (
                        self._variant_factory.parse(text_desc=text_desc, gene_symbol=row["gene"], refseq=transcript),
                        row["individual_id"],
                    )
                except ValueError as e:
                    logger.warning(f"Variant parsing failed: {e}")
                    return None, ""

            # For each paper, extract the (variant, subject)-specific key/value pairs into a new dict of dicts.
            variants = {_get_variant_key(row): {key: row.get(key, "") for key in TRUTHSET_VARIANT_KEYS} for row in rows}
            if (None, "") in variants:
                logger.warning("Dropping placeholder variants.")
                variants.pop((None, ""))

            if not variants:
                logger.warning(f"No valid variants for {paper_id}.")
                continue

            # Create a Paper object with the extracted fields.
            papers.add(Paper(id=paper_id, evidence=variants, **paper_data))

        return papers

    def search(self, query: str) -> Set[Paper]:
        """For the TruthsetFileLibrary, query is expected to be a gene symbol."""
        all_papers = self._load_truthset()

        # Filter to just the papers with variants that have evidence for the gene specified in the query.
        return {p for p in all_papers if query in {v[0].gene_symbol for v in p.evidence.keys()}}


class RemoteFileLibrary(IGetPapers):
    """A class for retrieving papers from PubMed."""

    def __init__(self, paper_client: IPaperLookupClient, max_papers: int = 5) -> None:
        """Initialize a new instance of the RemoteFileLibrary class.

        Args:
            paper_client (IPaperLookupClient): A class for searching and fetching papers.
            max_papers (int, optional): The maximum number of papers to retrieve. Defaults to 5.
        """
        self._paper_client = paper_client
        self._max_papers = max_papers

    def search(self, query: str) -> Set[Paper]:
        """Search for papers based on the given query.

        Args:
            query (str): The query to search for.

        Returns:
            Set[Paper]: The set of papers that match the query.
        """
        paper_ids = self._paper_client.search(query=query, max_papers=self._max_papers)
        papers = {paper for paper_id in paper_ids if (paper := self._paper_client.fetch(paper_id)) is not None}
        return papers
