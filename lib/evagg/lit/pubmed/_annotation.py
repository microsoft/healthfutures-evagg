from typing import Any

import requests

from lib.evagg.types import Paper

from .._interfaces import IAnnotateEntities


class PubtatorEntityAnnotator(IAnnotateEntities):
    def __init__(self) -> None:
        pass

    def annotate(self, paper: Paper) -> dict[str, Any]:
        """Annotate the paper with entities from PubTator.

        Returns paper annotations.
        """
        if "pmcid" not in paper.props:
            raise ValueError("Paper must have a PMC ID to be annotated by PubTator.")
        paper_id = paper.props["pmcid"]
        format = "json"

        response = requests.get(
            f"https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/bioc{format}?pmcids={paper_id}",
            timeout=10,
        )
        response.raise_for_status()

        # Can return a 200 with no valid result if the PMC ID is not found.
        if len(response.content) == 0:
            raise ValueError(f"PMC ID not found {paper_id}")
        return response.json()
