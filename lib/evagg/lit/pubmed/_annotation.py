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
        if "pmcid" not in paper.props:
            raise ValueError("Paper must have a PMC ID to be annotated by PubTator.")
        paper_id = paper.props["pmcid"]
        format = "json"

        r = requests.get(
            f"https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/bioc{format}?pmcids={paper_id}",
            timeout=10,
        )
        # TODO, handle errors.
        r.raise_for_status()
        return r.json()
