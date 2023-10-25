from typing import Any, Dict

import requests

from lib.evagg.types import Paper

from .._interfaces import IAnnotateEntities


class PubtatorEntityAnnotator(IAnnotateEntities):
    def __init__(self) -> None:
        pass

    def annotate(self, paper: Paper) -> Dict[str, Any]:
        """Annotate the paper with entities from PubTator.

        Returns paper annotations. If the paper does not have a PMC ID, a value error is raised. If the paper is not in
        PMC-OA then an empty dict is returned.
        """
        if "pmcid" not in paper.props or paper.props["pmcid"] is None:
            raise ValueError("Paper must have a PMC ID to be annotated by PubTator.")
        if "is_pmc_oa" not in paper.props or paper.props["is_pmc_oa"] == "False":
            # TODO, better way to check for PMC-OA?
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
            print(f"warning: empty response from PubTator for PMC {paper_id}")
            return {}
        return response.json()
