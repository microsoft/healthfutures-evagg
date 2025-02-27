import asyncio
import datetime
import logging
from typing import Any, Dict, Sequence

from lib.evagg.interfaces import IGetPapers
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import Paper

logger = logging.getLogger(__name__)


class ListLibrary(IGetPapers):
    """A class for fetching a specific set of papers from pubmed."""

    def __init__(
        self,
        paper_client: IPaperLookupClient,
        pmid_dict: Dict[str, list[str]],
    ) -> None:
        """Initialize a new instance of the ListLibrary class.

        Args:
            paper_client (IPaperLookupClient): A class for searching and fetching papers.
            pmid_dict (Dict[str, list[str]]): A dictionary mapping gene symbols to lists of PMIDs.
        """
        self._paper_client = paper_client
        self._pmid_dict = pmid_dict

    def get_papers(self, query: Dict[str, Any]) -> Sequence[Paper]:
        """Search for papers based on the given query.

        Args:
            query (Dict[str, Any]): The query to search for.

        Returns:
            Sequence[Paper]: The set of rare disease papers that match the query.
        """
        pmids_for_gene = self._pmid_dict.get(query["gene_symbol"], [])
        papers = [
            paper
            for paper_id in pmids_for_gene
            if (paper := self._paper_client.fetch(paper_id, include_fulltext=True)) is not None
            and paper.props["can_access"] is True
        ]
        import pdb

        pdb.set_trace()
        return papers
