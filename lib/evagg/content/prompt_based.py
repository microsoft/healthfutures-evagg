import json
import logging
import os
from typing import Any, Dict, List, Sequence

from lib.evagg.llm.openai import IOpenAIClient
from lib.evagg.ref import IVariantLookupClient
from lib.evagg.types import Paper

from ..interfaces import IExtractFields
from .interfaces import IFindVariantObservations, VariantObservation

logger = logging.getLogger(__name__)


class PromptBasedContentExtractor(IExtractFields):
    _SUPPORTED_FIELDS = {
        "gene",
        "paper_id",
        "hgvs_c",
        "hgvs_p",
        "individual_id",
        "phenotype",
        "zygosity",
        "variant_inheritance",
    }
    _PROMPTS = {
        "zygosity": os.path.dirname(__file__) + "/prompts/zygosity.txt",
        "variant_inheritance": os.path.dirname(__file__) + "/prompts/variant_inheritance.txt",
        "phenotype": os.path.dirname(__file__) + "/prompts/phenotype.txt",
    }

    def __init__(
        self,
        fields: Sequence[str],
        llm_client: IOpenAIClient,
        observation_finder: IFindVariantObservations,
        variant_lookup_client: IVariantLookupClient,
    ) -> None:
        self._fields = fields
        self._llm_client = llm_client
        self.observation_finder = observation_finder
        self._variant_lookup_client = variant_lookup_client

    def _excerpt_from_mentions(self, mentions: Sequence[Dict[str, Any]]) -> str:
        return "\n\n".join([m["text"] for m in mentions])

    def extract(self, paper: Paper, query: str) -> Sequence[Dict[str, str]]:
        # Only process papers in PMC.
        if "pmcid" not in paper.props or paper.props["pmcid"] == "":
            return []

        # Find all the variant observations in the paper relating to the query
        observations = self.observation_finder.find_variant_observations(query, paper)

        logger.info(f"Found {len(observations)} topics in {paper.id}")

        # For each variant observation/field pair, extract the appropriate content.
        results: List[Dict[str, str]] = []

        for obs in observations:
            variant = obs.variant
            individual_id = obs.individual_id

            topic_content: Dict[str, str] = {}
            logger.info(f"Extracting fields for {individual_id}:{variant} from {paper.id}")

            for field in self._fields:
                if field not in self._SUPPORTED_FIELDS:
                    raise ValueError(f"Unsupported field: {field}")

                if field == "gene":
                    result = variant.gene_symbol
                elif field == "paper_id":
                    result = paper.id
                elif field == "individual_id":
                    result = individual_id
                elif field == "hgvs_c":
                    # TODO, hgvs_c conversion from variant
                    result = variant.hgvs_desc
                elif field == "hgvs_p":
                    # TODO, hgvs_p conversion from variant
                    result = variant.hgvs_desc
                else:
                    # TODO, Use of obs.variant.gene_symbol is potentially dangerous if the paper uses a synonym
                    # for the gene that we've normalized in the HGVSVariant. We should consider using the gene
                    # symbol as expressed in the source text here, or HGVSVariant should include a list of synonyms
                    # for the gene symbol.
                    params = {
                        "passage": "\n\n".join(obs.mentions),
                        "variant": ",".join(obs.variant_identifiers),
                        "gene": obs.variant.gene_symbol if obs.variant.gene_symbol else "unknown",
                    }

                    response = self._llm_client.chat_oneshot_file(
                        user_prompt_file=self._PROMPTS[field],
                        system_prompt="Extract field",
                        params=params,
                    )
                    raw = response.output
                    try:
                        result = json.loads(raw)[field]
                    except Exception:
                        result = "failed"
                topic_content[field] = result
            results.append(topic_content)
        return results
