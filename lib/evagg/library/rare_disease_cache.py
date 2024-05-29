import logging
from typing import Any, Dict, List, Sequence

from lib.evagg.llm import IPromptClient
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import Paper
from lib.evagg.utils.cache import ObjectFileCache

from .rare_disease import RareDiseaseFileLibrary

logger = logging.getLogger(__name__)


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
        example_types: Sequence[str] | None = None,
    ) -> None:
        super().__init__(paper_client, llm_client, allowed_categories, example_types)
        self._cache = ObjectFileCache[Sequence[Paper]](
            "RareDiseaseFileLibrary",
            serializer=RareDiseaseLibraryCached.serialize_paper_sequence,
            deserializer=RareDiseaseLibraryCached.deserialize_paper_sequence,
        )

    def get_papers(self, query: Dict[str, Any]) -> Sequence[Paper]:
        cache_key = f"get_papers_{query['gene_symbol']}"
        if papers := self._cache.get(cache_key):
            logger.info(f"Retrieved {len(papers)} papers from cache for {query['gene_symbol']}.")
            return papers
        papers = super().get_papers(query)
        self._cache.set(cache_key, papers)
        return papers
