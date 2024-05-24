import asyncio
import csv
import json
import logging
import os
import re
from collections import defaultdict
from datetime import date
from functools import cache
from typing import Any, Dict, List, Sequence, Set

from lib.evagg.llm import IPromptClient
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.svc import ObjectFileCache
from lib.evagg.types import ICreateVariants, Paper

from .disease_keywords import INCLUSION_KEYWORDS
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
    "functional_study",
    # "gnomad_frequency",
    "study_type",
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

            # Create an evidence dictionary from the variant/patient-specific columns.
            evidence = {key: row.get(key, "") for key in TRUTHSET_EVIDENCE_KEYS}
            # Add a unique identifier for this combination of paper, variant, and individual ID.
            evidence["pub_ev_id"] = f"{paper.id}:{variant.hgvs_desc}:{row['individual_id']}".replace(" ", "")
            paper.evidence[(variant, row["individual_id"])] = evidence

        return paper

    @cache
    def _load_truthset(self) -> Set[Paper]:
        row_count = 0
        # Group the rows by paper ID.
        paper_groups = defaultdict(list)
        with open(self._file_path) as tsvfile:
            header = [h.strip() for h in tsvfile.readline().split("\t")]
            reader = csv.reader(tsvfile, delimiter="\t")
            for line in reader:
                fields = dict(zip(header, [field.strip() for field in line]))
                paper_groups[fields["paper_id"]].append(fields)
                row_count += 1

        logger.info(f"Loaded {row_count} rows with {len(paper_groups)} papers from {self._file_path}.")
        # Process each paper row group into a Paper object with truthset evidence filled in.
        papers = {self._process_paper(paper_id, rows) for paper_id, rows in paper_groups.items()}
        # Make sure that each evidence truthset row is unique across the truthset.
        assert len({ev["pub_ev_id"] for p in papers for ev in p.evidence.values()}) == row_count
        return papers

    def get_papers(self, query: Dict[str, Any]) -> Sequence[Paper]:
        """For the TruthsetFileLibrary, query is expected to be a gene symbol."""
        all_papers = self._load_truthset()

        if gene_symbol := query.get("gene_symbol"):
            # Filter to just the papers with variants that have evidence for the gene specified in the query.
            return [p for p in all_papers if gene_symbol in {v[0].gene_symbol for v in p.evidence.keys()}]
        return []


class RareDiseaseFileLibrary(IGetPapers):
    """A class for fetching and categorizing disease papers from PubMed."""

    CATEGORIES = ["rare disease", "other"]

    def __init__(
        self,
        paper_client: IPaperLookupClient,
        llm_client: IPromptClient,
        allowed_categories: Sequence[str] | None = None,
        example_types: Sequence[str] | None = None,
    ) -> None:
        """Initialize a new instance of the RareDiseaseFileLibrary class.

        Args:
            paper_client (IPaperLookupClient): A class for searching and fetching papers.
            llm_client (IPromptClient): A class to leverage LLMs to filter to the right papers.
            allowed_categories (Sequence[str], optional): The categories of papers to allow. Defaults to "rare disease".
            example_types (Sequence[str], optional): The types of examples to use in few shot setting. These can be
            positive or positive and negative examples. Default is just positive.
        """
        # TODO: go back and incorporate the idea of paper_types that can be passed into RareDiseaseFileLibrary,
        # so that the user of this class can specify which types of papers they want to filter for.
        self._paper_client = paper_client
        self._llm_client = llm_client
        self._allowed_categories = allowed_categories if allowed_categories is not None else ["rare disease"]
        # Allowed categories should be a subset of or equal to possible CATEGORIES, otherwise raise exception and halt
        if not set(self._allowed_categories).issubset(set(self.CATEGORIES)):
            raise ValueError(
                "Allowed categories must be a subset of or equal to the possible categories: 'rare disease' or 'other'."
            )
        self._example_types = example_types if example_types is not None else ["positive"]

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
        # If the paper doesn't fit in the other categories, add it to the other category.
        return "other"

    async def _get_llm_category(self, paper: Paper, gene: str) -> str:
        """Categorize papers based on LLM prompts."""
        # Load the few shot examples
        unique_file_name, _ = self._load_few_shot_examples(paper, gene, "few_shot")

        parameters = {
            "abstract": paper.props.get("abstract") or "no abstract",
            "title": paper.props.get("title") or "no title",
        }

        # Few shot examples embedded into paper finding classification prompt
        response = await self._llm_client.prompt_file(
            user_prompt_file=unique_file_name,
            system_prompt="Extract field",
            params=parameters,
            prompt_settings={"prompt_tag": "paper_category", "temperature": 0.8},
        )

        if isinstance(response, str):
            result = response
        else:
            logger.warning(f"LLM failed to return a valid categorization response for {paper.id}: {response}")

        if result in self.CATEGORIES:
            return result

        return "other"

    def _check_gene_in_string(self, text: str, gene: str) -> bool:
        lines = text.split("\n")
        for line in lines:
            if line.startswith("Gene: "):
                if gene in line[6:]:
                    return True
        return False

    def _replace_cluster_with_gene(self, examples: str, examples_bkup: str, gene: str) -> str:
        # Split into clusters
        clusters = re.split(r"(?=Gene: )", examples)
        clusters_bkup = re.split(r"(?=Gene: )", examples_bkup)

        # Iterate over clusters
        for i, cluster in enumerate(clusters):
            if self._check_gene_in_string(cluster, gene):
                # Replace the entire paper in the cluster
                papers = cluster.split("\n")
                papers_bkup = clusters_bkup[i].split("\n")
                start_index = next(
                    (j for j, paper in enumerate(papers) if self._check_gene_in_string(paper, gene)), None
                )
                if start_index is not None:
                    end_index = next(
                        (j for j in range(start_index + 1, len(papers)) if papers[j].startswith("Gene: ")), len(papers)
                    )
                    papers[start_index:end_index] = papers_bkup[start_index:end_index]
                clusters[i] = "\n".join(papers)

        # Join clusters back into a single string
        return "".join(clusters)

    def _load_few_shot_examples(self, paper: Paper, gene: str, method: str) -> List[str]:
        # Positive few shot examples
        with open("lib/evagg/content/prompts/few_shot_pos_examples.txt", "r") as filep:
            pos_file_content = filep.read()

        # Positive few shot examples backup if a gene in this file overlaps with the query paper gene, will pull
        # another from same cluster
        with open("lib/evagg/content/prompts/few_shot_pos_examples_bkup.txt", "r") as filepb:
            pos_file_content_bkup = filepb.read()

        if "negative" in self._example_types:
            # Negative few shot examples
            with open("lib/evagg/content/prompts/few_shot_neg_examples.txt", "r") as filen:
                neg_file_content = filen.read()

            # Negative few shot examples backup if a gene in this file overlaps with the query paper gene, will pull
            # another from same cluster
            with open("lib/evagg/content/prompts/few_shot_neg_examples_bkup.txt", "r") as filenb:
                neg_file_content_bkup = filenb.read()

        few_shot_phrases = (
            "\n\nBelow are several few shot examples of papers that are classified as 'rare disease'. "
            "These are in no particular order:\n"
            f"{self._replace_cluster_with_gene(pos_file_content, pos_file_content_bkup, gene)}\n"
        )

        if "negative" in self._example_types:
            few_shot_phrases += (
                "\nBelow are several few shot examples of papers that are classified as 'other'. "
                "These are in no particular order:\n"
                f"{self._replace_cluster_with_gene(neg_file_content, neg_file_content_bkup, gene)}\n"
            )

        # Read in paper_finding_*.txt and append the few shot examples
        with open(
            os.path.join(os.path.dirname(__file__), "content", "prompts", f"paper_finding_{method}.txt"), "r"
        ) as f:
            file_content = f.read()

        # Append few_shot_phrases
        file_content += few_shot_phrases

        # Generate a unique file name based on paper.id (PMID)
        unique_file_name = os.path.join(
            os.path.dirname(__file__),
            "content",
            "prompts",
            f"paper_finding_{method}_{paper.id.replace('pmid:', '')}.txt",
        )

        # Write the content to the unique file
        with open(unique_file_name, "w") as f:
            f.write(file_content)

        return [unique_file_name, few_shot_phrases]

    async def _get_paper_categorizations(self, paper: Paper, gene: str) -> str:
        """Categorize papers with multiple strategies and return the counts of each category."""
        # Categorize the paper by both keyword and LLM prompt.
        keyword_cat = self._get_keyword_category(paper)
        llm_cat = await self._get_llm_category(paper, gene)

        # If the keyword and LLM categories agree, just return that category.
        if keyword_cat == llm_cat:
            paper.props["disease_category"] = keyword_cat
            return keyword_cat

        counts: Dict[str, int] = {}

        llm_tiebreakers = await asyncio.gather(
            self._get_llm_category_few_shot(paper, gene), self._get_llm_category_few_shot(paper, gene)
        )

        # Otherwise it's conflicting - run the LLM prompt two more times and accumulate all the results.
        tiebreakers = await asyncio.gather(self._get_llm_category(paper, gene), self._get_llm_category(paper, gene))
        for category in [keyword_cat, llm_cat, *tiebreakers]:
            counts[category] = counts.get(category, 0) + 1
        assert len(counts) > 1 and sum(counts.values()) == 4

        best_category = max(counts, key=lambda k: counts[k])
        assert best_category in self.CATEGORIES and counts[best_category] < 4
        # Mark as conflicting if the best category has a low count.
        if counts[best_category] < 3:
            best_category = "conflicting"

        paper.props["disease_categorizations"] = counts
        paper.props["disease_category"] = best_category
        return best_category

    async def _get_all_papers(self, query: Dict[str, Any]) -> Sequence[Paper]:
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
        logger.info("Input query:", query)

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
            and paper.props["fulltext_xml"] is not None
        ]
        logger.info(f"Categorizing {len(papers)} papers for {query['gene_symbol']}.")

        await asyncio.gather(*[self._get_paper_categorizations(paper, query["gene_symbol"]) for paper in papers])
        return papers

    def get_papers(self, query: Dict[str, Any]) -> Sequence[Paper]:
        """Search for papers based on the given query.

        Args:
            query (Dict[str, Any]): The query to search for.

        Returns:
            Sequence[Paper]: The set of rare disease papers that match the query.
        """
        all_papers = asyncio.run(self._get_all_papers(query))
        return list(filter(lambda p: p.props["disease_category"] in self._allowed_categories, all_papers))


class RareDiseaseLibraryCached(RareDiseaseFileLibrary):
    """A class for fetching and categorizing disease papers from PubMed backed by a file-persisted cache."""

    @classmethod
    def serialize_paper_sequence(cls, papers: Sequence[Paper]) -> List[Dict[str, Any]]:
        return [paper.props for paper in papers]

    @classmethod
    def deserialize_paper_sequence(cls, data: List[Dict[str, Any]]) -> Sequence[Paper]:
        return [Paper(**paper) for paper in data]

    def __init__(
        self,
        paper_client: IPaperLookupClient,
        llm_client: IPromptClient,
        allowed_categories: Sequence[str] | None = None,
    ) -> None:
        super().__init__(paper_client, llm_client, allowed_categories)
        self._cache = ObjectFileCache[Sequence[Paper]](
            "RareDiseaseFileLibrary",
            serializer=RareDiseaseLibraryCached.serialize_paper_sequence,
            deserializer=RareDiseaseLibraryCached.deserialize_paper_sequence,
        )

    def get_papers(self, query: Dict[str, Any]) -> Sequence[Paper]:
        cache_key = f"get_papers_{query['gene_symbol']}"
        if papers := self._cache.get(cache_key):
            return papers
        papers = super().get_papers(query)
        self._cache.set(cache_key, papers)
        return papers
