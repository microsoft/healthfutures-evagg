import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Sequence

from lib.evagg.interfaces import IGetPapers
from lib.evagg.llm import IPromptClient
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import Paper
from lib.evagg.utils import PROMPT_DIR

from .disease_keywords import INCLUSION_KEYWORDS
from .few_shot_examples import NEGATIVE_EXAMPLES, POSITIVE_EXAMPLES

logger = logging.getLogger(__name__)


class RareDiseaseFileLibrary(IGetPapers):
    """A class for fetching and categorizing disease papers from PubMed."""

    CATEGORIES = ["rare disease", "other"]

    def __init__(
        self,
        paper_client: IPaperLookupClient,
        llm_client: IPromptClient,
        allowed_categories: Sequence[str] | None = None,
    ) -> None:
        """Initialize a new instance of the RareDiseaseFileLibrary class.

        Args:
            paper_client (IPaperLookupClient): A class for searching and fetching papers.
            llm_client (IPromptClient): A class to leverage LLMs to filter to the right papers.
            allowed_categories (Sequence[str], optional): The categories of papers to allow. Defaults to "rare disease".
        """
        # TODO: go back and incorporate the idea of paper_types that can be passed into RareDiseaseFileLibrary,
        # so that the user of this class can specify which types of papers they want to filter for.
        self._paper_client = paper_client
        self._llm_client = llm_client
        self._allowed_categories = allowed_categories if allowed_categories else ["rare disease"]
        # Allowed categories must be a subset of or equal to possible CATEGORIES.
        if not set(self._allowed_categories).issubset(set(self.CATEGORIES)):
            raise ValueError(f"Invalid category set: {self._allowed_categories}")

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
        # Build a list of examples using the text of whichever example in each pair doesn't match the current gene.
        positive_examples = [e[0].text if e[0].gene != gene else e[1].text for e in POSITIVE_EXAMPLES]
        negative_examples = [e[0].text if e[0].gene != gene else e[1].text for e in NEGATIVE_EXAMPLES]

        parameters = {
            "abstract": paper.props.get("abstract") or "no abstract",
            "title": paper.props.get("title") or "no title",
            "positive_examples": "".join(positive_examples),
            "negative_examples": "".join(negative_examples),
        }

        # Few shot examples embedded into paper finding classification prompt
        response = await self._llm_client.prompt_file(
            user_prompt_file=os.path.join(PROMPT_DIR, "paper_category.txt"),
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

    async def _get_paper_categorizations(self, paper: Paper, gene: str) -> str:
        """Categorize papers with multiple strategies and return the counts of each category."""
        # Categorize the paper by both keyword and LLM prompt.
        keyword_cat = self._get_keyword_category(paper)
        llm_cat = await self._get_llm_category(paper, gene)

        # If the keyword and LLM categories agree, just return that category.
        if keyword_cat == llm_cat:
            paper.props["disease_category"] = keyword_cat
            paper.props["disease_categorizations"] = "{}"
            return keyword_cat

        counts: Dict[str, int] = {}
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

        paper.props["disease_category"] = best_category
        paper.props["disease_categorizations"] = json.dumps(counts)
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
        # Rationalize the optional parameters.
        if ("max_date" in query or "date_type" in query) and "min_date" not in query:
            raise ValueError("A min_date is required when max_date or date_type is provided.")
        if "min_date" in query:
            params["min_date"] = query["min_date"]
            params["date_type"] = query.get("date_type", "pdat")
        if "max_date" in query:
            params["max_date"] = query["max_date"]
        if "retmax" in query:
            params["retmax"] = query["retmax"]

        # Perform the search for papers
        paper_ids = self._paper_client.search(**params)
        logger.info(f"Fetching {len(paper_ids)} papers for {query['gene_symbol']}.")

        # Extract the paper content that we care about (e.g. title, abstract, PMID, etc.)
        papers = [
            paper
            for paper_id in paper_ids
            if (paper := self._paper_client.fetch(paper_id, include_fulltext=True)) is not None
            and paper.props["fulltext_xml"] is not None
        ]
        logger.info(f"Categorizing {len(papers)} papers with full text for {query['gene_symbol']}.")

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
