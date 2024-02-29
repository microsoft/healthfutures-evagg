import logging
from typing import Any, Dict, Sequence, Tuple

from lib.evagg.llm import IPromptClient, OpenAIClient
from lib.evagg.ref import IPaperLookupClient, NcbiLookupClient
from lib.evagg.svc import CosmosCachingWebClient, get_dotenv_settings
from lib.evagg.types import HGVSVariant, Paper

logger = logging.getLogger(__name__)


def make_topic_finder() -> "ObservationFinder":
    # ncbi_settings:
    #   di_factory: lib.evagg.svc.get_dotenv_settings
    #   filter_prefix: "NCBI_EUTILS_"
    # web_client:
    #   di_factory: lib.evagg.svc.CosmosCachingWebClient
    #   cache_settings:
    #     di_factory: lib.evagg.svc.get_dotenv_settings
    #     filter_prefix: "EVAGG_CONTENT_CACHE_"
    # ncbi_client:
    #   di_factory: lib.evagg.ref.NcbiLookupClient
    #   web_client: "{{web_client}}"
    #   settings: "{{ncbi_settings}}"

    return ObservationFinder(
        OpenAIClient(get_dotenv_settings(filter_prefix="AZURE_OPENAI_")),
        NcbiLookupClient(
            web_client=CosmosCachingWebClient(cache_settings=get_dotenv_settings(filter_prefix="EVAGG_CONTENT_CACHE_")),
            settings=get_dotenv_settings(filter_prefix="NCBI_EUTILS_"),
        ),
    )


class ObservationFinder:
    def __init__(self, llm_client: IPromptClient, paper_lookup_client: IPaperLookupClient) -> None:
        self._llm_client = llm_client
        self._paper_lookup_client = paper_lookup_client

    def _find_individuals(self, full_text: str) -> Sequence[str]:
        response = self._llm_client.prompt_file(
            self._PROMPTS["individuals"],
            system_prompt=self._SYSTEM_PROMPT,
            params=params,
            prompt_settings={"prompt_tag": "individuals"},
        )

        try:
            result = json.loads(response)
        except Exception:
            logger.warning(f"Failed to parse response from LLM: {response}")
            return []
        return result

    def find_observations(self, query: str, paper: Paper) -> Dict[Tuple[HGVSVariant, str], Sequence[Any]]:
        """Identify all observations relevant to `query` in `paper`.

        `query` should be a gene_symbol. `paper` is the paper to search for relevant observations. Paper must be in the
        PMC-OA dataset and have license terms that permit derivative works based on current restrictions.

        Observations are logically "clinical" observations of a variant in a human, thus this function returns a dict
        keyed by tuples of variants and string representations of the individual in which that variant was observed. The
        values in this dictionary are a collection of mentions relevant to this observation throughout the paper.
        """
        # Obtain the full-text of the paper.
        full_text = self._paper_lookup_client.full_text(paper)
        if full_text is None:
            logger.warning(f"Skipping {paper.id} because full text could not be retrieved")
            return {}

        # Determine all of the human subjects specifically referred to in the paper, if any.
        individuals = self._find_individuals(full_text)

        # Determine all of the genetic variants matching `query`
        variants = self._find_variants(query, full_text)

        # If there are both variants and human subjects, build a mapping between the two,
        # if there are only variants and no individuals, set all individuals to unknown,
        # if there are no variants (regardless of individuals), then there are no observations to report.
        if variants and individuals:
            return self._find_observations(variants, individuals, full_text)
        elif variants:
            return self._find_observations(variants, ["unknown"], full_text)
        else:
            return {}

        # Gather observations
        return {}
