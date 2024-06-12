import logging
from typing import Any, Dict, Sequence

from lib.evagg.types import Paper
from lib.evagg.utils.cache import ObjectFileCache

from .prompt_based import PromptBasedContentExtractor

logger = logging.getLogger(__name__)


class PromptBasedContentExtractorCache(PromptBasedContentExtractor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        use_previous_cache = kwargs.pop("use_previous_cache", None)
        self._cache = ObjectFileCache[Sequence[Dict[str, str]]](
            "PromptBasedContentExtractor",
            use_previous_cache=use_previous_cache,
        )
        super().__init__(*args, **kwargs)

    def extract(self, paper: Paper, gene_symbol: str) -> Sequence[Dict[str, str]]:
        cache_key = f"extract_{paper.props['pmid']}_{gene_symbol}"
        if obs := self._cache.get(cache_key):
            logger.info(f"Retrieved {len(obs)} field extractions from cache for {paper.id}/{gene_symbol}.")
            return obs
        obs = super().extract(paper, gene_symbol)
        self._cache.set(cache_key, obs)
        return obs
