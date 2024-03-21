import csv
import json
import logging
import os
import re
from collections import defaultdict
from datetime import date
from functools import cache
from typing import Any, Dict, Sequence, Set, Tuple

from lib.evagg.llm import IPromptClient
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.svc.logging import LogProvider
from lib.evagg.types import HGVSVariant, ICreateVariants, Paper

from .interfaces import IGetPapers

logger = logging.getLogger(__name__)
# TODO: ways to improve: process full text of paper to filter to rare disease papers when PMC OA


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
        if not query["gene_symbol"]:
            raise NotImplementedError("Minimum requirement to search is to input a gene symbol.")

        # Get gene term
        term = query["gene_symbol"]
        logger.info("\nFinding papers for gene:", term, "...")

        paper_ids = self._paper_client.search(query=term)
        papers = {paper for paper_id in paper_ids if (paper := self._paper_client.fetch(paper_id)) is not None}
        return papers


class RareDiseaseFileLibrary(IGetPapers, IPaperLookupClient, IPromptClient):
    """A class for filtering to rare disease papers from PubMed."""

    def __init__(self, paper_client: IPaperLookupClient, llm_client: IPromptClient) -> None:
        """Initialize a new instance of the RemoteFileLibrary class.

        Args:
            paper_client (IPaperLookupClient): A class for searching and fetching papers.
            llm_client (IPromptClient): A class to leveral LLMs to filter to the right papers.
        """
        self._paper_client = paper_client
        self._llm_client = llm_client

    def get_papers(self, query: Dict[str, Any]) -> Set[Paper]:
        """Search for papers based on the given query.

        Args:
            query (Dict[str, Any]): The query to search for.

        Returns:
            Set[Paper]: The set of papers that match the query.
        """
        if not query["gene_symbol"]:
            raise NotImplementedError("Minimum requirement to search is to input a gene symbol.")

        # Get gene term
        term = query["gene_symbol"]
        logger.info("\nFinding papers for gene:", term, "...")

        # Find paper IDs
        paper_ids = self._partition_search_query(query)

        # Extract the paper content that we care about (e.g. title, abstract, PMID, etc.)
        papers = {paper for paper_id in paper_ids if (paper := self._paper_client.fetch(paper_id)) is not None}

        # Call private function to filter for rare disease papers
        rare_disease_papers, _, _ = self._filter_rare_disease_papers(papers)

        # TODO: figure out how best to integrate these outputs (rare_disease_papers and llm_rare_disease_papers)
        llm_rare_disease_papers, _, _ = self._prompts_for_rare_disease_papers(papers)

        print("get_papers, RESULTS", len(llm_rare_disease_papers))

        return rare_disease_papers

    def get_all_papers(self, query: Dict[str, Any]) -> Tuple[Set[Paper], Set[Paper], Set[Paper], Set[Paper]]:
        """Search for papers based on the given query.

        Args:
            query (Dict[str, Any]): The query to search for.

        Returns:
           Tuple of Set[Paper]s: The sets of papers that match the query, across categories and overall (rare disease,
           non-rare disease, other, and the union).
        """
        if not query["gene_symbol"]:
            raise NotImplementedError("Minimum requirement to search is to input a gene symbol.")

        paper_ids = self._partition_search_query(query)

        # Extract the paper content that we care about (e.g. title, abstract, PMID, etc.)
        papers = {paper for paper_id in paper_ids if (paper := self._paper_client.fetch(paper_id)) is not None}

        # Call private function to filter for rare disease papers
        # TODO: fix
        # rare_disease_papers, non_rare_disease_papers, other_papers = self._filter_rare_disease_papers(papers)

        llm_rare_disease_papers, llm_non_rare_disease_papers, llm_other_papers = self._prompts_for_rare_disease_papers(
            papers
        )
        print("RESULTS", len(llm_rare_disease_papers), len(llm_non_rare_disease_papers), len(llm_other_papers))

        return llm_rare_disease_papers, llm_non_rare_disease_papers, llm_other_papers, papers

    def _prompts_for_rare_disease_papers(self, papers: Set[Paper]) -> Tuple[Set[Paper], Set[Paper], Set[Paper]]:
        """Apply LLM prompts to categorize papers into rare, non-rare, or other group."""
        # TODO: this should likely only re-check non-rare and other papers as the base implementation.
        #       call a filter and a prompt function on one runthrough of each paper, rather than
        #       iterate through papers twice?

        LogProvider(prompts_to_console=False)

        prompt = {
            "paper_category": os.path.dirname(__file__) + "/content/prompts/paper_finding.txt",
        }

        rare_disease_papers: Set[Paper] = set()
        non_rare_disease_papers: Set[Paper] = set()
        other_papers: Set[Paper] = set()

        for paper in papers:
            paper_pmid = paper.props.get("pmid", "Unknown")
            print("PMID:", paper_pmid)
            paper_title = paper.props.get("title", "Unknown")
            paper_abstract = paper.props.get("abstract", "Unknown")

            # print("paper_title", paper_title)
            # print("paper_abstract", paper_abstract)
            if paper_title is None:
                paper_title = "Unknown"
            if paper_abstract is None:
                paper_abstract = "Unknown"

            params = {"abstract": paper_abstract, "title": paper_title}

            response = self._llm_client.prompt_file(
                user_prompt_file=prompt["paper_category"],
                system_prompt="Extract field",
                params=params,
                prompt_settings={"prompt_tag": "paper_category"},
            )

            try:
                result = json.loads(response)["paper_category"]
            except Exception:
                result = "failed"  # TODO: how to handle this?

            print("llm result", result)
            # Categorize the paper based on LLM result
            if result == "rare disease":
                rare_disease_papers.add(paper)
            elif result == "non-rare disease":
                non_rare_disease_papers.add(paper)
            elif result == "other":
                other_papers.add(paper)
            elif result == "failed":
                logger.warning(f"Failed to categorize paper: {paper.id}")
            else:
                raise ValueError(f"Unexpected result: {result}")
        return rare_disease_papers, non_rare_disease_papers, other_papers

    def _partition_search_query(self, query: Dict[str, Any]) -> Sequence[str]:
        """Partition the query and run search to generate the paper IDs list for a given gene."""
        # Get gene term
        term = query["gene_symbol"]
        logger.info("\nFinding papers for gene:", term, "...")

        # Find paper IDs
        min_date = query.get("min_date", None)
        max_date = query.get("max_date", None)
        date_type = query.get("date_type", None)
        retmax = query.get("retmax", None)

        # If the most basic query is provided, search for papers with just the gene symbol
        if term and not min_date and not max_date and not date_type and not retmax:
            paper_ids = self._paper_client.search(query=term)
        # If min_date is the only extra parameter provided from this query,
        # max_date should be today's date and date_type should be "pdat".
        elif (
            min_date and not max_date and not date_type
        ):  # NLM requires min and max date: https://www.nlm.nih.gov/dataguide/eutilities/utilities.html
            max_date = date.today().strftime("%Y/%m/%d")
            date_type = "pdat"
            paper_ids = self._paper_client.search(query=term, min_date=min_date, date_type=date_type)  # type: ignore
        # If min_date and date_type are the only extra parameters provided from this query,
        # max_date should be today's date.
        elif min_date and date_type and not max_date:
            max_date = date.today().strftime("%Y/%m/%d")
            paper_ids = self._paper_client.search(
                query=term, min_date=min_date, max_date=max_date, date_type=date_type  # type: ignore
            )
        # If date_type is the only parameter provided,
        # we need to raise an error to say that a min_date and/or max_date should be provided.
        elif date_type and not min_date and not max_date:
            raise ValueError("A min_date and max_date should be provided when date_type is provided")
        # If max_date is provided but min_date and date_type are not provided,
        # throw an error to say that min_date and date_type should be provided.
        elif max_date and not min_date and not date_type:
            raise ValueError("A min_date (and optionally date_type) should be provided when max_date is provided")
        elif max_date and date_type and not min_date:
            raise ValueError("A min_date should be provided when max_date and date_type are provided")
        # If min and max dates are provided but not date_type, set to default: publication date.
        elif min_date and max_date and not date_type:
            date_type = "pdat"
            paper_ids = self._paper_client.search(
                query=term,
                min_date=min_date,
                max_date=max_date,
                date_type=date_type,  # type: ignore
            )
        else:
            paper_ids = self._paper_client.search(
                query=term,
                min_date=min_date,
                max_date=max_date,  # type: ignore
                date_type=date_type,  # type: ignore
                retmax=retmax,  # default retmax (i.e. max_papers) is 20
            )

        return paper_ids

    def _filter_rare_disease_papers(self, papers: Set[Paper]) -> Tuple[Set[Paper], Set[Paper], Set[Paper]]:
        """Filter papers to only include those that are related to rare diseases.

        Args:
            papers (Set[Paper]): The set of papers to filter.

        Returns:
            Set[Paper]: The set of papers that are related to rare diseases.
        """
        rare_disease_papers: Set[Paper] = set()
        non_rare_disease_papers: Set[Paper] = set()
        other_papers: Set[Paper] = set()

        # Iterate through each paper and filter into 1 of 3 categories based on title and abstract
        for paper in papers:
            paper_title = paper.props.get("title", None)
            paper_abstract = paper.props.get("abstract", None)

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
                "variant of uncertain significance",
                "variants of uncertain significance",
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
                "digenic",  # not meaningful in plural form
                "familial",  # not meaningful in plural form
                "structural variant",
                "somatic",  # not meaningful in plural form
                "somatic cancer",
                "cancer",
                "tumor",
                "CNV",
                "carcinoma",
                "copy number variant",
            ]

            exclusion_keywords = exclusion_keywords + [word + "s" for word in exclusion_keywords]

            # include in rare disease category
            if paper_title is not None and (
                any(keyword in paper_title.lower() for keyword in inclusion_keywords if not keyword.startswith("-"))
                or any(
                    re.search(f"{keyword[1:]}$", word)  # remove the "-" from "-" keywords
                    for keyword in inclusion_keywords
                    if keyword.startswith("-")  # match end of all words with these keywords
                    for word in paper_title.lower().split()
                )
            ):
                rare_disease_papers.add(paper)
            elif paper_abstract is not None and (
                any(keyword in paper_abstract.lower() for keyword in inclusion_keywords if not keyword.startswith("-"))
                or any(
                    re.search(f"{keyword[1:]}$", word)
                    for keyword in inclusion_keywords
                    if keyword.startswith("-")
                    for word in paper_abstract.lower().split()
                )
            ):
                rare_disease_papers.add(paper)
            # exclude from rare disease category, include in non-rare disease category
            elif paper_title is not None and (
                any(keyword in paper_title.lower() for keyword in exclusion_keywords if not keyword.startswith("-"))
                or any(
                    re.search(f"{keyword[1:]}$", word)
                    for keyword in exclusion_keywords
                    if keyword.startswith("-")
                    for word in paper_title.lower().split()
                )
            ):
                non_rare_disease_papers.add(paper)
            elif paper_abstract is not None and (
                any(keyword in paper_abstract.lower() for keyword in exclusion_keywords if not keyword.startswith("-"))
                or any(
                    re.search(f"{keyword[1:]}$", word)
                    for keyword in exclusion_keywords
                    if keyword.startswith("-")
                    for word in paper_abstract.lower().split()
                )
            ):
                non_rare_disease_papers.add(paper)
            # exclude from rare disease category, exclude from non-rare disease category, include in other category
            else:
                other_papers.add(paper)

            # include in rare disease category
            if paper_title is not None:
                if any(
                    keyword in paper_title.lower() for keyword in inclusion_keywords if not keyword.startswith("-")
                ) or any(
                    paper_title.lower().endswith(keyword[1:])
                    for keyword in inclusion_keywords
                    if keyword.startswith("-")
                ):
                    rare_disease_papers.add(paper)
            elif paper_abstract is not None:
                if any(
                    keyword in paper_abstract.lower() for keyword in inclusion_keywords if not keyword.startswith("-")
                ) or any(
                    paper_abstract.lower().endswith(keyword[1:])
                    for keyword in inclusion_keywords
                    if keyword.startswith("-")
                ):
                    rare_disease_papers.add(paper)
            # exclude from rare disease category, include in non-rare disease category
            elif paper_title is not None:
                if any(
                    keyword in paper_title.lower() for keyword in exclusion_keywords if not keyword.startswith("-")
                ) or any(
                    paper_title.lower().endswith(keyword[1:])
                    for keyword in exclusion_keywords
                    if keyword.startswith("-")
                ):
                    non_rare_disease_papers.add(paper)
            elif paper_abstract is not None:
                if any(
                    keyword in paper_abstract.lower() for keyword in exclusion_keywords if not keyword.startswith("-")
                ) or any(
                    paper_abstract.lower().endswith(keyword[1:])
                    for keyword in exclusion_keywords
                    if keyword.startswith("-")
                ):
                    non_rare_disease_papers.add(paper)
            # exclude from rare disease category, exclude from non-rare disease category, include in other category
            else:
                other_papers.add(paper)

            # Exclude papers that are not written in English by scanning the title or abstract
            # TODO: Implement this

            # Exclude papers that only describe animal models and do not have human data
            # TODO: Implement this

            # TODO: add in GPT-4 code

        # Output the number of papers in each category
        logger.info("Rare Disease Papers: ", len(rare_disease_papers))
        logger.info("Non-Rare Disease Papers: ", len(non_rare_disease_papers))
        logger.info("Other Papers: ", len(other_papers))

        return rare_disease_papers, non_rare_disease_papers, other_papers
