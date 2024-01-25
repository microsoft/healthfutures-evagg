import logging
from typing import Any, Dict

import requests

from lib.evagg.types import Paper

from ..interfaces import IAnnotateEntities

logger = logging.getLogger(__name__)


class PubtatorEntityAnnotator(IAnnotateEntities):
    def __init__(self) -> None:
        pass

    def annotate(self, paper: Paper) -> Dict[str, Any]:
        """Annotate the paper with entities from PubTator.

        Returns paper annotations. If the paper is not in PMC-OA then an empty dict is returned.
        """
        if not paper.props.get("pmcid") or not paper.props.get("is_pmc_oa"):
            logger.warning(f"Cannot annotate, paper {paper.id} is not in PMC-OA")
            return {}

        paper_id = paper.props["pmcid"]
        format = "json"
        response = requests.get(
            f"https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/bioc{format}?pmcids={paper_id}",
            timeout=10,
        )
        response.raise_for_status()

        # Can return a 200 with no valid result if the PMC ID is not found, this can happen if the paper is in PMC but
        # not PMC-OA.
        if len(response.content) == 0:
            logger.warning(f"Empty response from PubTator for PMC {paper_id}")
            return {}
        return response.json()
