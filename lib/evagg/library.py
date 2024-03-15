import csv
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from functools import cache
from typing import Any, Dict, List, Sequence, Set, Tuple

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

    def get_papers(self, query: Dict[str, Any]) -> Set[Paper]:
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
                paper_id = fields.get("doi") or fields.get("pmid") or fields.get("pmcid") or "MISSING_ID"
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

    def get_papers(self, query: Dict[str, Any]) -> Set[Paper]:
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

    def get_papers(self, query: Dict[str, Any]) -> Set[Paper]:
        """Search for papers based on the given query.

        Args:
            query (IPaperQuery): The query to search for.
        Returns:
            Set[Paper]: The set of papers that match the query.
        """
        if len(query) > 1:
            raise NotImplementedError("Multiple term extraction not yet implemented.")
        term = next(iter(query))
        paper_ids = self._paper_client.search(query=term, max_papers=self._max_papers)
        papers = {paper for paper_id in paper_ids if (paper := self._paper_client.fetch(paper_id)) is not None}
        return papers


class RareDiseaseFileLibrary(IGetPapers):
    """A class for filtering to rare disease papers from PubMed."""

    def __init__(
        self,
        paper_client: IPaperLookupClient,
        max_papers: int = 5,
    ) -> None:
        """Initialize a new instance of the RemoteFileLibrary class.
        Args:
            paper_client (IPaperLookupClient): A class for searching and fetching papers.
            max_papers (int, optional): The maximum number of papers to retrieve. Defaults to 5.
        """
        self._paper_client = paper_client
        self._max_papers = max_papers

    def get_papers(self, query: Dict[str, Any]):
        """Search for papers based on the given query.

        Args:
            query (str): The query to search for.

        Returns:
            Set[Paper]: The set of papers that match the query.
        """
        if not query["gene_symbol"]:
            raise NotImplementedError("Minimum requirement to search is to input a gene symbol.")

        # Get gene term
        term = query["gene_symbol"]
        logger.info("\nFinding papers for gene:", term, "...")

        # Find paper IDs
        paper_ids = self._paper_client.search(
            query=term,
            max_papers=self._max_papers,  # ,
            min_date=query["min_date"],
            max_date=query["max_date"],
            date_type=query["date_type"],
        )

        # Extract the paper content that we care about (e.g. title, abstract, PMID, etc.)
        papers = {paper for paper_id in paper_ids if (paper := self._paper_client.fetch(paper_id)) is not None}

        # Call private function to filter for rare disease papers
        # "_, _" are non_rare_disease_papers and other_papers, respectively
        rare_disease_papers, count_r_d_papers, non_rare_disease_papers, other_papers = self._filter_rare_disease_papers(
            papers
        )  # TODO: We only need count_r_d_papers, non_rare_disease_papers and other_papers for benchmarking, so is
        # returning them the right way to handle this? Otherwise I can call the search() and
        # _filter_rare_disease_papers() directly.

        if count_r_d_papers == 0:
            rare_disease_papers = set()
        return rare_disease_papers, count_r_d_papers, non_rare_disease_papers, other_papers, papers

    def _filter_rare_disease_papers(self, papers: Set[Paper]):
        """Filter papers to only include those that are related to rare diseases.
        Args:
            papers (Set[Paper]): The set of papers to filter.
        Returns:
            Set[Paper]: The set of papers that are related to rare diseases.
        """

        rare_disease_papers = set()
        non_rare_disease_papers = set()
        other_papers = set()

        for paper in papers:
            paper_title = paper.props.get("title", "Unknown")
            paper_abstract = paper.props.get("abstract", "Unknown")

            inclusion_keywords = [
                "variant",
                "rare disease",
                "rare variant",
                "disorder",
                "syndrome",
                "-emia",
                "-cardia",
                "-phagia",
                "pathogenic",
                "benign",
                "inherited cancer",
                "germline",
            ]

            inclusion_keywords = inclusion_keywords + [word + "s" for word in inclusion_keywords]

            inclusion_keywords_odd_plurals = [
                "-lepsy",
                "-lepsies",
                "-pathy",
                "-pathies",
                "-osis",
                "-oses",
                "variant of unknown significance",
                "variants of unknown significance",
                "variant of uncertain significance" "variants of uncertain significance",
            ]

            inclusion_keywords_no_plural = [
                "mendelian",
                "monogenic",
                "monogenicity",
                "monoallelic",
                "syndromic",
                "inherited",
                "hereditary",
                "dominant",
                "recessive",
                "de novo",
                "VUS",
                "disease causing",
            ]

            inclusion_keywords = inclusion_keywords + inclusion_keywords_odd_plurals + inclusion_keywords_no_plural

            exclusion_keywords = [
                "digenic",
                "familial",
                "structural variant",
                "structural variants",
                "somatic",
                "somatic cancer",
                "somatic cancers",
                "cancer",
                "cancers",
                "CNV",
                "CNVs",
                "copy number variant",
                "copy number variants",
            ]

            # include
            if paper_title is not None and any(keyword in paper_title.lower() for keyword in inclusion_keywords):
                rare_disease_papers.add(paper)
            elif paper_abstract is not None and any(
                keyword in paper_abstract.lower() for keyword in inclusion_keywords
            ):
                rare_disease_papers.add(paper)
            # exclude
            elif paper_title is not None and any(keyword in paper_title.lower() for keyword in exclusion_keywords):
                non_rare_disease_papers.add(paper)
            elif paper_abstract is not None and any(
                keyword in paper_abstract.lower() for keyword in exclusion_keywords
            ):
                non_rare_disease_papers.add(paper)
            # other
            else:
                other_papers.add(paper)

            # Exclude papers that are not written in English by scanning the title or abstract
            # TODO: Implement this

            # Exclude papers that only describe animal models and do not have human data
            # TODO: Implement this

        logger.info("Rare Disease Papers: ", len(rare_disease_papers))
        logger.info("Non-Rare Disease Papers: ", len(non_rare_disease_papers))
        logger.info("Other Papers: ", len(other_papers))

        # Check if rare_disease_papers is empty or if non_rare_disease_papers is empty
        cnt_r_d_p = 1
        if len(rare_disease_papers) == 0:
            cnt_r_d_p = 0
            rare_disease_papers = Set[Paper]
        if len(non_rare_disease_papers) == 0:
            non_rare_disease_papers = Set[Paper]

        return rare_disease_papers, cnt_r_d_p, non_rare_disease_papers, other_papers
