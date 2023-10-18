import json
from typing import Any, Dict, List, Sequence

from lib.evagg.lit import IFindVariantMentions
from lib.evagg.sk import ISemanticKernelClient
from lib.evagg.types import IPaperQuery, Paper

from .._interfaces import IExtractFields


class SemanticKernelContentExtractor(IExtractFields):
    _SUPPORTED_FIELDS = {"gene", "paper_id", "hgvsc", "hgvsp", "phenotype", "zygosity", "inheritance"}

    def __init__(
        self, fields: Sequence[str], sk_client: ISemanticKernelClient, mention_finder: IFindVariantMentions
    ) -> None:
        self._fields = fields
        self._sk_client = sk_client
        self._mention_finder = mention_finder

    def _excerpt_from_mentions(self, mentions: Sequence[Dict[str, Any]]) -> str:
        return "\n\n".join([m["text"] for m in mentions])

    def extract(self, paper: Paper, query: IPaperQuery) -> Sequence[Dict[str, str]]:
        # Find all the variant mentions in the paper relating to the query.
        variant_mentions = self._mention_finder.find_mentions(query, paper)

        # For each variant/field pair, extract the appropriate content.
        results: List[Dict[str, str]] = []

        for variant_id in variant_mentions.keys():
            mentions = variant_mentions[variant_id]
            variant_results: Dict[str, str] = {"variant": variant_id}

            # Simplest thing we can think of is to just concatenate all the chunks.
            paper_excerpts = self._excerpt_from_mentions(mentions)
            context_variables = {
                "input": paper_excerpts,
                "variant": variant_id,
                "gene": mentions[0].get("gene_symbol", "unknown"),  # Mentions should never be empty.
            }
            for field in self._fields:
                if field not in self._SUPPORTED_FIELDS:
                    raise ValueError(f"Unsupported field: {field}")

                if field == "gene":
                    result = mentions[0].get("gene_symbol", "unknown")  # Mentions should never be empty.
                elif field == "paper_id":
                    result = paper.id
                elif field == "hgvsc":
                    result = f"{variant_id}"  # TODO
                elif field == "hgvsp":
                    result = f"{variant_id}"  # TODO
                else:
                    raw = self._sk_client.run_completion_function(
                        skill="content", function=field, context_variables=context_variables
                    )
                    result = json.loads(raw)[field]
                variant_results[field] = result
            results.append(variant_results)
        return results
