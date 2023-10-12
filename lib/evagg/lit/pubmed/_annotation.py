from typing import Any

import requests

from lib.evagg import Paper

from .._interfaces import IAnnotateEntities


class PubtatorEntityAnnotator(IAnnotateEntities):
    def __init__(self) -> None:
        pass

    def annotate(self, paper: Paper) -> dict[str, Any]:
        """Annotate the paper with entities from PubTator.

        Returns paper annotations.
        """
        format = "json"
        paper_id = paper.pmcid

        r = requests.get(
            f"https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/bioc{format}?pmcids={paper_id}",
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
