import csv
import json
import logging
import os
import re
from collections import defaultdict
from datetime import date
from functools import cache
from typing import Any, Dict, List, Sequence, Set, Tuple

from lib.evagg.llm import IPromptClient
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.svc.logging import LogProvider
from lib.evagg.types import HGVSVariant, ICreateVariants, Paper

from .interfaces import IGetPapers

logger = logging.getLogger(__name__)
# TODO: ways to improve:
#       - process full text of paper to filter to rare disease papers when PMC OA


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
        # Get gene term
        term = query["gene_symbol"]
        logger.info("\nFinding papers for gene:", term, "...")

        paper_ids = self._paper_client.search(query=term)
        papers = {paper for paper_id in paper_ids if (paper := self._paper_client.fetch(paper_id)) is not None}
        return papers


class RareDiseaseFileLibrary(IGetPapers):
    """A class for filtering to rare disease papers from PubMed."""

    def __init__(
        self,
        paper_client: IPaperLookupClient,
        llm_client: IPromptClient,
        # TODO: go back and incorporate the idea of paper_types that can be passed into RareDiseaseFileLibrary,
        # so that the user of this class can specify which types of papers they want to filter for.
    ) -> None:
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
            Set[Paper]: The set of rare disease papers that match the query.
        """
        print("Query:", query)
        return self.get_all_papers(query)[0]

    def get_all_papers(
        self, query: Dict[str, Any]
    ) -> Tuple[Set[Paper], Set[Paper], Set[Paper], Set[Paper], Set[Paper], List]:
        """Search for papers based on the given query.

        Args:
            query (Dict[str, Any]): The query to search for.

        Returns:
           Tuple of Set[Paper]s: The sets of papers that match the query, across categories and overall (rare disease,
           non-rare disease, other, and the union).
        """
        if not query.get("gene_symbol"):
            raise ValueError("Minimum requirement to search is to input a gene symbol.")

        paper_ids = self._partition_search_query(query)

        # Extract the paper content that we care about (e.g. title, abstract, PMID, etc.)
        papers = {paper for paper_id in paper_ids if (paper := self._paper_client.fetch(paper_id)) is not None}

        (
            rare_disease_tie_break,
            non_rare_disease_tie_break,
            other_tie_break,
            discordant_human_in_loop,
            counts_discordant_hil,
        ) = self._tie_breaking_assessment(papers)

        return (
            rare_disease_tie_break,
            non_rare_disease_tie_break,
            other_tie_break,
            papers,
            discordant_human_in_loop,
            counts_discordant_hil,
        )

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

        # Check for missing parameters and set defaults
        if (
            min_date and not max_date
        ):  # If min_date is the only extra parameter provided from this query,max_date should be today's date and
            # date_type will default to "pdat". NLM requires min and max date:
            # https://www.nlm.nih.gov/dataguide/eutilities/utilities.html
            max_date = date.today().strftime("%Y/%m/%d")
        if (
            min_date and max_date and not date_type
        ):  # If min and max dates are provided but not date_type, set to default: publication date.
            date_type = "pdat"
        if date_type and not min_date and not max_date:
            raise ValueError("A min_date and optionally max_date should be provided when date_type is provided")
        if max_date and not min_date:
            raise ValueError("A min_date should be provided when max_date is provided")

        # Perform the search
        params = {
            "query": term,
            "min_date": min_date,
            "max_date": max_date,
            "date_type": date_type,
            "retmax": retmax,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        # Perform the search for papers
        paper_ids = self._paper_client.search(**params)

        return paper_ids

    def _contains_keywords(self, text: str, keywords: List[str]) -> bool:  # type: ignore
        """Check if a text contains any of the keywords."""
        return any(keyword in text.lower() for keyword in keywords if not keyword.startswith("-")) or any(
            re.search(f"{keyword[1:]}$", word)
            for keyword in keywords
            if keyword.startswith("-")
            for word in text.lower().split()
        )

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

        # Keywords
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

        # Iterate through each paper and filter into 1 of 3 categories based on title and abstract
        for paper in papers:
            paper_title = paper.props.get("title", "")
            paper_abstract = paper.props.get("abstract", "")

            # Check if the paper should be included in the rare disease category
            if (paper_title and self._contains_keywords(paper_title, inclusion_keywords)) or (
                paper_abstract and self._contains_keywords(paper_abstract, inclusion_keywords)
            ):
                rare_disease_papers.add(paper)
            # Check if the paper should be included in the non-rare disease category
            elif (paper_title and self._contains_keywords(paper_title, exclusion_keywords)) or (
                paper_abstract and self._contains_keywords(paper_abstract, exclusion_keywords)
            ):
                non_rare_disease_papers.add(paper)
            # If the paper doesn't fit in the other categories, add it to the other category
            else:
                other_papers.add(paper)

            # TODO: Exclude papers that are not written in English by scanning the title or abstract

            # TODO: Exclude papers that only describe animal models and do not have human data

        # Output the number of papers in each category
        logger.info("Rare Disease Papers: ", len(rare_disease_papers))
        logger.info("Non-Rare Disease Papers: ", len(non_rare_disease_papers))
        logger.info("Other Papers: ", len(other_papers))

        return rare_disease_papers, non_rare_disease_papers, other_papers

    def _prompts_for_rare_disease_papers(self, papers: Set[Paper]) -> Tuple[Set[Paper], Set[Paper], Set[Paper]]:
        """Apply LLM prompts to categorize papers into rare, non-rare, or other group."""
        LogProvider(prompts_to_console=False)

        prompt = {
            "paper_category": os.path.join(os.path.dirname(__file__), "content", "prompts", "paper_finding.txt"),
        }

        paper_categories: Dict[str, Set] = {
            "rare disease": set(),
            "non-rare disease": set(),
            "other": set(),
        }

        for paper in papers:
            paper_pmid = paper.props.get("pmid", "Unknown")
            print("PMID:", paper_pmid)
            paper_title = paper.props.get("title", "Unknown")
            paper_abstract = paper.props.get("abstract", "Unknown")

            params = {"abstract": paper_abstract or "Unknown", "title": paper_title or "Unknown"}

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

            # Categorize the paper based on LLM result
            if result in paper_categories:
                paper_categories[result].add(paper)
            elif result == "failed":
                logger.warning(f"Failed to categorize paper: {paper.id}")
            else:
                raise ValueError(f"Unexpected result: {result}")

        return paper_categories["rare disease"], paper_categories["non-rare disease"], paper_categories["other"]

    def _tie_breaking_assessment(self, papers: Set[Paper]) -> Tuple[set, set, set, set, List]:
        categories = ["rare_disease", "non_rare_disease", "other"]
        tie_breaks: Dict[str, Set] = {category: set() for category in categories}
        discordant = set()
        discordant_human_in_loop = set()

        filter_papers = self._filter_rare_disease_papers(papers)
        llm_papers = self._prompts_for_rare_disease_papers(papers)

        # Compare filter and llm classifications (rare, non, other) and see if any discordant papers
        for i, (category, filter_set) in enumerate(zip(categories, filter_papers)):
            llm_set = llm_papers[i]
            tie_breaks[category] = filter_set.intersection(llm_set)
            discordant.update(filter_set.difference(llm_set))
            discordant.update(llm_set.difference(filter_set))

        # If discordant papers, run 2 llm methods to see if we can break the tie, otherwise add to discordant list for
        # manual review
        count_discordant_hil = []
        if discordant:
            llm_papers_2 = self._prompts_for_rare_disease_papers(discordant)
            llm_papers_3 = self._prompts_for_rare_disease_papers(discordant)

            for paper in discordant:
                paper_counts = [
                    sum(
                        paper in paper_set
                        for paper_set in [
                            filter_papers[i],
                            llm_papers[i],
                            llm_papers_2[i],
                            llm_papers_3[i],
                        ]
                    )
                    for i in range(len(categories))
                ]
                counts = dict(zip(categories, paper_counts))
                max_category = max(counts, key=lambda k: counts[k], default=None)
                if max_category and counts[max_category] >= 3:
                    tie_breaks[max_category].add(paper)
                else:
                    discordant_human_in_loop.add(paper)
                    count_discordant_hil.append(counts)

        return (
            tie_breaks["rare_disease"],
            tie_breaks["non_rare_disease"],
            tie_breaks["other"],
            discordant_human_in_loop,
            count_discordant_hil,
        )
