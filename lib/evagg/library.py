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
from lib.evagg.types import HGVSVariant, ICreateVariants, Paper

from .disease_keywords import EXCLUSION_KEYWORDS, INCLUSION_KEYWORDS
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

    def get_papers(self, query: Dict[str, Any]) -> Sequence[Paper]:
        # Dummy implementation that returns all papers regardless of query.
        all_papers = list(self._load().values())
        return all_papers


# These are the columns in the truthset that are specific to the paper.
TRUTHSET_PAPER_KEYS = ["paper_id", "pmid", "pmcid", "paper_title", "is_pmc_oa", "license"]
TRUTHSET_PAPER_KEYS_MAPPING = {"paper_title": "title"}
# These are the columns in the truthset that are specific to the variant.
TRUTHSET_VARIANT_KEYS = [
    "gene",
    "transcript",
    "paper_variant",
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
    _paper_client: IPaperLookupClient

    def __init__(self, file_path: str, variant_factory: ICreateVariants, paper_client: IPaperLookupClient) -> None:
        self._file_path = file_path
        self._variant_factory = variant_factory
        self._paper_client = paper_client

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

            for row in rows:
                if "is_pmc_oa" in row:
                    row["is_pmc_oa"] = row["is_pmc_oa"].lower() == "true"  # type: ignore

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
            if paper_id is not None and paper_id.startswith("pmid:"):
                pmid = paper_id[5:]
                paper = self._paper_client.fetch(pmid, include_fulltext=True)
                if paper:
                    # Compare and potentially add in truthset data that we don't get from the paper client.
                    for key in TRUTHSET_PAPER_KEYS:
                        if key == "paper_id":
                            continue
                        mapped_key = TRUTHSET_PAPER_KEYS_MAPPING.get(key, key)
                        if mapped_key in paper.props:
                            if paper.props[mapped_key] != paper_data[key]:
                                logger.warning(
                                    f"Paper field mismatch: {key}/{mapped_key} ({paper_data[key]} vs"
                                    f" {paper.props[mapped_key]})."
                                )
                        else:
                            logger.error(f"Adding {mapped_key}:{paper_data[key]} to paper props.")
                            paper.props[mapped_key] = paper_data[key]
                if not paper:
                    logger.warning(f"Failed to fetch paper with PMID {pmid}.")
                    paper = Paper(id=paper_id, evidence=variants, **paper_data)
                else:
                    # Add in evidence.
                    paper.evidence = variants
            else:
                logger.warning("Paper ID does not appear to be a pmid, cannot fetch paper content.")
                paper = Paper(id=paper_id, evidence=variants, **paper_data)

            papers.add(paper)

        return papers

    def get_papers(self, query: Dict[str, Any]) -> Sequence[Paper]:
        """For the TruthsetFileLibrary, query is expected to be a gene symbol."""
        all_papers = self._load_truthset()

        if gene_symbol := query.get("gene_symbol"):
            # Filter to just the papers with variants that have evidence for the gene specified in the query.
            return [p for p in all_papers if gene_symbol in {v[0].gene_symbol for v in p.evidence.keys()}]
        return []


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

    def get_papers(self, query: Dict[str, Any]) -> Sequence[Paper]:
        """Search for papers based on the given query.

        Args:
            query (IPaperQuery): The query to search for.

        Returns:
            Sequence[Paper]: The set of papers that match the query.
        """
        # Get gene term
        if not query.get("gene_symbol"):
            raise ValueError("Minimum requirement to search is to input a gene symbol.")

        paper_ids = self._paper_client.search(query=query["gene_symbol"])
        papers = [paper for paper_id in paper_ids if (paper := self._paper_client.fetch(paper_id)) is not None]
        return papers


class RareDiseaseFileLibrary(IGetPapers):
    """A class for fetching and categorizing disease papers from PubMed."""

    CATEGORIES = ["rare disease", "non-rare disease", "other"]  # "conflicting" is another potential category.

    def __init__(
        self,
        paper_client: IPaperLookupClient,
        llm_client: IPromptClient,
        allowed_categories: Sequence[str] | None = None,
        require_full_text: bool = False,
    ) -> None:
        """Initialize a new instance of the RareDiseaseFileLibrary class.

        Args:
            paper_client (IPaperLookupClient): A class for searching and fetching papers.
            llm_client (IPromptClient): A class to leverage LLMs to filter to the right papers.
            allowed_categories (Sequence[str], optional): The categories of papers to allow. Defaults to "rare disease".
            require_full_text (bool, optional): Whether to require full text for the paper. Defaults to False.
        """
        # TODO: go back and incorporate the idea of paper_types that can be passed into RareDiseaseFileLibrary,
        # so that the user of this class can specify which types of papers they want to filter for.
        self._paper_client = paper_client
        self._llm_client = llm_client
        self._allowed_categories = allowed_categories if allowed_categories is not None else ["rare disease"]
        self._require_full_text = require_full_text

    def _get_keyword_category(self, paper: Paper) -> str:
        """Categorize papers based on keywords in the title and abstract."""
        title = paper.props.get("title") or ""
        abstract = paper.props.get("abstract") or ""

        # TODO: Exclude papers that are not written in English by scanning the title or abstract
        # TODO: Exclude papers that only describe animal models and do not have human data

        def _has_keywords(text: str, keywords: List[str]) -> bool:
            """Check if a text contains any of the keywords."""
            if not text:
                return False

            full_text = text.lower()
            split_text = full_text.split()

            for keyword in keywords:
                if keyword.startswith("-"):
                    if any(re.search(f"{keyword[1:]}$", word) for word in split_text):
                        return True
                else:
                    if keyword in full_text:
                        return True
            return False

        # Check if the paper should be included in the rare disease category.
        if _has_keywords(title, INCLUSION_KEYWORDS) or _has_keywords(abstract, INCLUSION_KEYWORDS):
            return "rare disease"
        # Check if the paper should be included in the non-rare disease category.
        if _has_keywords(title, EXCLUSION_KEYWORDS) or _has_keywords(abstract, EXCLUSION_KEYWORDS):
            return "non-rare disease"
        # If the paper doesn't fit in the other categories, add it to the other category.
        return "other"

    def _get_paper_texts(self, paper: Paper) -> Dict[str, Any]:
        texts: Dict[str, Any] = {}
        texts["full_text"] = "\n".join(paper.props.get("full_text_sections", []))
        texts["tables"] = "\n".join(self._get_tables_from_paper(paper).values())
        return texts

    def _get_tables_from_paper(self, paper: Paper) -> Dict[str, str]:
        tables = {}
        root = paper.props.get("full_text_xml")
        if root is not None:
            for passage in root.findall("./passage"):
                if bool(passage.findall("infon[@key='section_type'][.='TABLE']")):
                    id = passage.find("infon[@key='id']").text
                    if not id:
                        logger.error("No id for table, using None as key")
                    tables[id] = tables.get(id, "") + "\n" + passage.find("text").text
        return tables

    def _get_llm_category(self, paper: Paper) -> str:
        paper_finding_txt = (
            "paper_finding.txt" if paper.props.get("full_text_xml") is None else "paper_finding_full_text.txt"
        )
        parameters = (
            self._get_paper_texts(paper)
            if paper_finding_txt == "paper_finding_full_text.txt"
            else {
                "abstract": paper.props.get("abstract") or "no abstract",
                "title": paper.props.get("title") or "no title",
            }
        )
        response = self._llm_client.prompt_file(
            user_prompt_file=os.path.join(os.path.dirname(__file__), "content", "prompts", paper_finding_txt),
            system_prompt="Extract field",
            params=parameters,
            prompt_settings={"prompt_tag": "paper_category", "temperature": 0.8},
        )
        try:
            result = json.loads(response).get("paper_category", response)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from LLM: {response}")
            return "other"
        if result in self.CATEGORIES:
            return result
        logger.warning(f"LLM failed to return a valid categorization response for {paper.id}: {response}")
        return "other"

    def _get_paper_categorizations(self, paper: Paper) -> Dict[str, int]:
        """Categorize papers with multiple strategies and return the counts of each category."""
        # Categorize the paper by both keyword and LLM prompt.
        keyword_cat = self._get_keyword_category(paper)
        llm_cat = self._get_llm_category(paper)

        # If the keyword and LLM categories agree, just return that category.
        if keyword_cat == llm_cat:
            return {keyword_cat: 2}

        counts: Dict[str, int] = {}
        # Otherwise run the LLM prompt two more times and accumulate all the results.
        for category in [keyword_cat, llm_cat, self._get_llm_category(paper), self._get_llm_category(paper)]:
            counts[category] = counts.get(category, 0) + 1
        assert sum(counts.values()) == 4
        return counts

    def _get_all_papers(self, query: Dict[str, Any]) -> Sequence[Paper]:
        """Search for papers based on the given query.

        Args:
            query (Dict[str, Any]): The query to search for.

        Returns:
            Sequence[Paper]: The papers that match the query, tagged with a `disease_category` property representing
            what disease type it is judged to reference (rare disease, non-rare disease, other, and conflicting). If
            the paper is tagged as conflicting, it will also have a `disease_categorizations` property that shows the
            counts of the categorizations.
        """
        if not query.get("gene_symbol"):
            raise ValueError("Minimum requirement to search is to input a gene symbol.")
        params = {"query": query["gene_symbol"]}

        # Rationalize the optional parameters.
        if ("max_date" in query or "date_type" in query) and "min_date" not in query:
            raise ValueError("A min_date is required when max_date or date_type is provided.")
        if "min_date" in query:
            params["min_date"] = query["min_date"]
            params["max_date"] = query.get("max_date", date.today().strftime("%Y/%m/%d"))
            params["date_type"] = query.get("date_type", "pdat")
        if "retmax" in query:
            params["retmax"] = query["retmax"]

        # Perform the search for papers
        paper_ids = self._paper_client.search(**params)

        # Extract the paper content that we care about (e.g. title, abstract, PMID, etc.)
        papers = [
            paper
            for paper_id in paper_ids
            if (paper := self._paper_client.fetch(paper_id, include_fulltext=True)) is not None
        ]

        if self._require_full_text:
            papers = [p for p in papers if p.props.get("full_text_xml")]

        logger.warning(f"Categorizing {len(papers)} papers for {query['gene_symbol']}.")

        # Categorize the papers.
        for paper in papers:
            categories = self._get_paper_categorizations(paper)
            best_category = max(categories, key=lambda k: categories[k])
            assert best_category in self.CATEGORIES and categories[best_category] < 4

            # If there are multiple categories and the best one has a low count, mark it conflicting.
            if len(categories) > 1:
                # Always keep categorizations if there's more than one category.
                paper.props["disease_categorizations"] = categories
                # Mark as conflicting if the best category has a low count.
                if categories[best_category] < 3:
                    best_category = "conflicting"

            paper.props["disease_category"] = best_category

        return papers

    def get_papers(self, query: Dict[str, Any]) -> Sequence[Paper]:
        """Search for papers based on the given query.

        Args:
            query (Dict[str, Any]): The query to search for.

        Returns:
            Sequence[Paper]: The set of rare disease papers that match the query.
        """
        return list(
            filter(lambda p: p.props["disease_category"] in self._allowed_categories, self._get_all_papers(query))
        )
