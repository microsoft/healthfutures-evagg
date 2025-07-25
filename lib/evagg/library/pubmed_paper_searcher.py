import datetime
import logging
from collections.abc import Sequence
from typing import Any

from lib.evagg.ref.ncbi import NcbiClientBase
from lib.evagg.utils import IWebContentClient

from .interfaces import ISearchPapers

logger = logging.getLogger(__name__)


class PubMedPaperSearcher(NcbiClientBase, ISearchPapers):
    """Search for papers using PubMed via NCBI E-utilities."""

    def __init__(self, web_client: IWebContentClient, settings: dict[str, str] | None = None) -> None:
        """Initialize PubMed paper searcher.

        Args:
            web_client: Web client for making HTTP requests.
            settings: Optional NCBI API settings (API key, email).
        """
        super().__init__(web_client, settings)

    async def search_papers(self, gene_symbol: str, query_params: dict[str, Any]) -> Sequence[str]:
        """Search for paper IDs related to gene_symbol using PubMed.

        This is extracted from RareDiseaseFileLibrary._get_all_papers() with minimal changes
        to ensure functional equivalence.

        Args:
            gene_symbol: Gene symbol to search for.
            query_params: Additional query parameters (min_date, max_date, retmax, etc.).

        Returns:
            Sequence of paper IDs (PMIDs) matching the search criteria.
        """
        if not gene_symbol:
            raise ValueError("Minimum requirement to search is to input a gene symbol.")

        params: dict[str, Any] = {"query": f"{gene_symbol} pubmed pmc open access[filter]"}
        # Rationalize the optional parameters.
        if ("max_date" in query_params or "date_type" in query_params) and "min_date" not in query_params:
            raise ValueError("A min_date is required when max_date or date_type is provided.")
        if "min_date" in query_params:
            params["mindate"] = query_params["min_date"]
            params["date_type"] = query_params.get("date_type", "pdat")
            params["maxdate"] = query_params.get("max_date", datetime.datetime.now().strftime("%Y/%m/%d"))
        if "retmax" in query_params:
            params["retmax"] = query_params["retmax"]

        # Perform the search for papers using NCBI E-utilities
        query = params.pop("query")
        retmax = params.get("retmax")

        root = self._esearch(db="pubmed", term=query, sort="relevance", **params)
        paper_ids = [id.text for id in root.findall("./IdList/Id") if id.text]

        logger.info(f"Found {len(paper_ids)} paper IDs for {gene_symbol}.")

        if retmax is not None and len(paper_ids) == retmax:
            logger.warning(
                f"Reached the maximum number of papers ({retmax}) for {gene_symbol}. This may cause "
                "an incomplete result, with not all papers matching the gene and date range being processed."
            )

        return paper_ids
