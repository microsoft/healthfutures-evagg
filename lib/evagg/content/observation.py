import os
from typing import Sequence

from lib.evagg.llm.openai import IOpenAIClient
from lib.evagg.ref import IAnnotateEntities
from lib.evagg.types import Paper

from .interfaces import IFindVariantObservations, VariantObservation


class VariantObservationFinder(IFindVariantObservations):
    """Find variant observations in a paper."""

    _VARIANT_FINDING_PROMPT = os.path.dirname(__file__) + "/prompts/observation/list_variants.txt"

    def __init__(self, llm_client: IOpenAIClient, annotator: IAnnotateEntities) -> None:
        self._llm_client = llm_client
        self._annotator = annotator

    def _get_full_text(self, paper: Paper) -> str:
        # Get the full text of a paper.
        anno = self._annotator.annotate(paper)

    def find_variant_observations(self, query: str, paper: Paper) -> Sequence[VariantObservation]:
        """Find variant observations relevant to query that are mentioned in `paper`.

        Returns a list of VariantObservations.
        """

        # Get the full text of the paper.
        full_text = self._get_full_text(paper)

        variant_list_raw = self._llm_client.chat_oneshot_file(self._VARIANT_FINDING_PROMPT, None, params)
        return []
