"""Paper library that returns predefined lists of papers for specific genes."""

import logging
from typing import Any, Dict, List, Sequence

from lib.evagg.interfaces import IGetPapers
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import Paper

logger = logging.getLogger(__name__)


class PaperListLibrary(IGetPapers):
    """A library that returns predefined papers for specific gene queries.

    This implementation allows configuring specific PMIDs to return for
    each gene symbol, bypassing the normal paper search process.
    """

    def __init__(self, paper_client: IPaperLookupClient, gene_pmid_mapping: Dict[str, List[str]]) -> None:
        """Initialize the PaperListLibrary.

        Args:
            paper_client: Client for fetching paper details by PMID
            gene_pmid_mapping: Dictionary mapping gene symbols to lists of PMIDs
        """
        self._paper_client = paper_client
        self._gene_pmid_mapping = gene_pmid_mapping

    def get_papers(self, query: Dict[str, Any]) -> Sequence[Paper]:
        """Get papers for a gene query based on predefined PMID mappings.

        Args:
            query: Query dictionary containing at minimum a 'gene_symbol' key

        Returns:
            Sequence of Paper objects for the gene, or empty sequence if no mapping exists

        Raises:
            ValueError: If gene_symbol is not provided in the query
        """
        if not query.get("gene_symbol"):
            raise ValueError("Minimum requirement to search is to input a gene symbol.")

        gene_symbol = query["gene_symbol"]

        # Get the PMIDs for this gene from the mapping
        pmids = self._gene_pmid_mapping.get(gene_symbol, [])

        if not pmids:
            logger.info(f"No PMIDs configured for gene symbol: {gene_symbol}")
            return []

        logger.info(f"Fetching {len(pmids)} predefined papers for {gene_symbol}")

        # Fetch papers for each PMID
        papers = []
        for pmid in pmids:
            paper = self._paper_client.fetch(pmid, include_fulltext=True)
            if paper is not None:
                papers.append(paper)
            else:
                logger.warning(f"Could not fetch paper for PMID: {pmid}")

        logger.info(f"Successfully fetched {len(papers)} papers for {gene_symbol}")
        return papers
