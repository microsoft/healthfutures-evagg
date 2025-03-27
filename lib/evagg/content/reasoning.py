import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Sequence

from lib.evagg.llm import IPromptClient
from lib.evagg.types import Paper
from lib.evagg.utils.cache import ObjectFileCache

from ..interfaces import IExtractFields
from .fulltext import get_fulltext

logger = logging.getLogger(__name__)


def _get_prompt_file_path(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "prompts", f"{name}.txt")


class ReasoningContentExtractor(IExtractFields):
    _PROMPT_FIELDS = {
        "content": _get_prompt_file_path("content_reasoning"),
    }
    # Read the system prompt from file
    _SYSTEM_PROMPT = open(_get_prompt_file_path("system")).read()

    _DEFAULT_PROMPT_SETTINGS = {
        "prompt_tag": "content_reasoning",
        # Use for AOAI
        # "max_completion_tokens": 30000,
        # "reasoning_effort": "high",
        # "response_format": {"type": "json_object"},
        # "timeout": 300,
        # Use for foundry.
        "response_format": "json_object",
        "temperature": 0.7,
        ##"top_p": 0.95,
        "max_tokens": 30000,
    }

    def __init__(
        self,
        fields: Sequence[str],
        llm_client: IPromptClient,
        prompt_settings: Dict[str, Any] | None = None,
    ) -> None:
        self._fields = fields
        self._llm_client = llm_client
        self._instance_prompt_settings = (
            {**self._DEFAULT_PROMPT_SETTINGS, **prompt_settings} if prompt_settings else self._DEFAULT_PROMPT_SETTINGS
        )

    async def _run_json_prompt(
        self, prompt_filepath: str, params: Dict[str, str], prompt_settings: Dict[str, Any]
    ) -> Dict[str, Any]:

        prompt_settings = {**self._instance_prompt_settings, **prompt_settings}

        response = await self._llm_client.prompt_file(
            user_prompt_file=prompt_filepath,
            system_prompt=self._SYSTEM_PROMPT,
            params=params,
            prompt_settings=prompt_settings,
        )

        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse response from LLM to {prompt_filepath}: {response}")
            return {}

        return result

    async def _extract_all(self, paper: Paper, gene_symbol: str) -> List[Dict[str, str]]:
        # Paper text.
        fulltext = get_fulltext(paper.props["fulltext_xml"], exclude=["AUTH_CONT", "ACK_FUND", "COMP_INT", "REF"])

        # Run the main reasoning prompt and get back the base dictionary.
        params = {"gene": gene_symbol, "fulltext": fulltext}
        prompt_settings = {
            "prompt_tag": "content",
            "prompt_metadata": {"gene_symbol": gene_symbol, "paper_id": paper.id},
        }
        try:
            obs_dicts_raw = await self._run_json_prompt(self._PROMPT_FIELDS["content"], params, prompt_settings)
        except Exception as e:
            logger.error(f"Caught exception extracting content for {paper}: {e}")
            return []

        obs_dicts = obs_dicts_raw.get("observations", [])

        # Populate the lookup fields.
        for obs_dict in obs_dicts:
            obs_dict["id"] = (
                paper.id
                + "_"
                + gene_symbol
                + "_"
                + obs_dict["individual_id"].replace(" ", "")
                + "_"
                + obs_dict["hgvs_c"]
                if obs_dict["hgvs_c"]
                else obs_dict["hgvs_p"]
            )
            obs_dict["gene"] = gene_symbol
            obs_dict["paper_id"] = paper.id
            obs_dict["citation"] = paper.props["citation"]
            obs_dict["link"] = (
                "https://www.ncbi.nlm.nih.gov/pmc/articles/" + paper.props["pmcid"]
                if "pmcid" in paper.props
                else paper.props["link"]
            )
            obs_dict["paper_title"] = paper.props["title"]
            obs_dict["valid"] = True
            obs_dict["validation_error"] = ""

        return obs_dicts

    def extract(self, paper: Paper, gene_symbol: str) -> Sequence[Dict[str, str]]:
        if not paper.props.get("can_access", False):
            logger.warning(f"Skipping {paper.id} because it is not licensed for access")
            return []

        # Run the primary reasoning prompt.
        content = asyncio.get_event_loop().run_until_complete(self._extract_all(paper, gene_symbol))

        if not content:
            logger.info(f"No observations found in {paper.id} for {gene_symbol}")

        return content


class ReasoningContentExtractorCached(ReasoningContentExtractor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        use_previous_cache = kwargs.pop("use_previous_cache", None)
        self._cache = ObjectFileCache[Sequence[Dict[str, str]]](
            "ReasoningContentExtractor",
            use_previous_cache=use_previous_cache,
        )
        super().__init__(*args, **kwargs)

    def extract(self, paper: Paper, gene_symbol: str) -> Sequence[Dict[str, str]]:
        cache_key = f"extract_{paper.props['pmid']}_{gene_symbol}"
        if (obs := self._cache.get(cache_key)) is not None:
            logger.info(f"Retrieved {len(obs)} field extractions from cache for {paper.id}/{gene_symbol}.")
            return obs
        obs = super().extract(paper, gene_symbol)
        self._cache.set(cache_key, obs)
        return obs
