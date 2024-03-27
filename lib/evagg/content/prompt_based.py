import json
import logging
import os
from typing import Any, Dict, List, Sequence

from lib.evagg.llm import IPromptClient
from lib.evagg.types import Paper

from ..interfaces import IExtractFields
from .interfaces import IFindObservations

logger = logging.getLogger(__name__)


class PromptBasedContentExtractor(IExtractFields):
    _SUPPORTED_FIELDS = {
        "gene",
        "paper_id",
        "hgvs_c",
        "hgvs_p",
        "transcript",
        "individual_id",
        "phenotype",
        "zygosity",
        "variant_inheritance",
        "valid",
    }
    _PROMPTS = {
        "zygosity": os.path.dirname(__file__) + "/prompts/zygosity.txt",
        "variant_inheritance": os.path.dirname(__file__) + "/prompts/variant_inheritance.txt",
        "phenotype": os.path.dirname(__file__) + "/prompts/phenotype.txt",
    }

    def __init__(
        self,
        fields: Sequence[str],
        llm_client: IPromptClient,
        observation_finder: IFindObservations,
    ) -> None:
        self._fields = fields
        self._llm_client = llm_client
        self._observation_finder = observation_finder

    def _excerpt_from_mentions(self, mentions: Sequence[Dict[str, Any]]) -> str:
        return "\n\n".join([m["text"] for m in mentions])

    def extract(self, paper: Paper, gene_symbol: str) -> Sequence[Dict[str, str]]:
        # Only process papers in PMC-OA.
        if (
            "pmcid" not in paper.props
            or paper.props["pmcid"] == ""
            or "is_pmc_oa" not in paper.props
            or paper.props["is_pmc_oa"] is False
        ):
            logger.warning(f"Skipping {paper.id} because it is not in PMC-OA")
            return []

        # Find all the observations in the paper relating to the query.
        observations = self._observation_finder.find_observations(gene_symbol, paper)

        logger.info(f"Found {len(observations)} observations in {paper.id}")

        if not observations:
            logger.warning(f"No observations found in {paper.id}")
            return []

        # For each observation, extract the appropriate content.
        results: List[Dict[str, str]] = []

        for observation, mentions in observations.items():
            variant, individual = observation
            observation_results: Dict[str, str] = {}

            logger.info(f"Extracting fields for {observation} in {paper.id}")

            if not mentions:
                logger.warning(f"No mentions found for {observation} in {paper.id}")
                continue

            # Simplest thing we can think of is to just concatenate all the chunks.
            paper_excerpts = "\n\n".join(mentions)
            gene_symbol = gene_symbol

            for field in self._fields:
                if field not in self._SUPPORTED_FIELDS:
                    raise ValueError(f"Unsupported field: {field}")

                if field == "gene":
                    result = gene_symbol
                elif field == "paper_id":
                    result = paper.id
                elif field == "individual_id":
                    result = individual
                elif field == "hgvs_c":
                    result = variant.hgvs_desc if not variant.hgvs_desc.startswith("p.") else "NA"
                elif field == "hgvs_p":
                    if variant.protein_consequence:
                        result = variant.protein_consequence.hgvs_desc
                    else:
                        result = variant.hgvs_desc if variant.hgvs_desc.startswith("p.") else "NA"
                elif field == "transcript":
                    result = variant.refseq if variant.refseq else "unknown"
                elif field == "valid":
                    result = str(variant.valid)
                else:
                    # TODO, should be the original text representation of the variant from the paper. When we switch to
                    # actual mention objects, we can fix this.
                    params = {"passage": paper_excerpts, "variant": variant.__str__(), "gene": gene_symbol}

                    response = self._llm_client.prompt_file(
                        user_prompt_file=self._PROMPTS[field],
                        system_prompt="Extract field",
                        params=params,
                        prompt_settings={"prompt_tag": field},
                    )
                    raw = response
                    try:
                        result = json.loads(raw)[field]
                    except Exception:
                        result = "failed"
                observation_results[field] = result
            results.append(observation_results)
        return results
