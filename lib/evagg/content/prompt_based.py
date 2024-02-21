import json
import logging
import os
from typing import Any, Dict, List, Sequence

from lib.evagg.llm import IPromptClient
from lib.evagg.ref import IVariantLookupClient
from lib.evagg.types import Paper

from ..interfaces import IExtractFields
from .interfaces import IFindVariantMentions

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
        llm_client: IPromptClient,
        mention_finder: IFindVariantMentions,
        variant_lookup_client: IVariantLookupClient,
    ) -> None:
        self._fields = fields
        self._llm_client = llm_client
        self._mention_finder = mention_finder
        self._variant_lookup_client = variant_lookup_client

    def _excerpt_from_mentions(self, mentions: Sequence[Dict[str, Any]]) -> str:
        return "\n\n".join([m["text"] for m in mentions])

    def extract(self, paper: Paper, query: str) -> Sequence[Dict[str, str]]:
        # Only process papers in PMC.
        if "pmcid" not in paper.props or paper.props["pmcid"] == "":
            return []

        # Find all the variant mentions in the paper relating to the query.
        variant_mentions = self._mention_finder.find_mentions(query, paper)

        logger.info(f"Found {len(variant_mentions)} variant mentions in {paper.id}")

        # For each variant/field pair, extract the appropriate content.
        results: List[Dict[str, str]] = []

        for variant, mentions in variant_mentions.items():
            variant_results: Dict[str, str] = {}

            logger.info(f"Extracting fields for {variant} in {paper.id}")

            # Simplest thing we can think of is to just concatenate all the chunks.
            paper_excerpts = self._excerpt_from_mentions(mentions)
            gene_symbol = mentions[0].get("gene_symbol", "unknown")  # Mentions should never be empty.

            for field in self._fields:
                if field not in self._SUPPORTED_FIELDS:
                    raise ValueError(f"Unsupported field: {field}")

                if field == "gene":
                    result = gene_symbol
                elif field == "paper_id":
                    result = paper.id
                elif field == "individual_id":
                    result = "unknown"
                elif field == "hgvs_c":
                    result = variant.hgvs_desc
                elif field == "hgvs_p":
                    result = variant.hgvs_desc
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
                variant_results[field] = result
            results.append(variant_results)
        return results
