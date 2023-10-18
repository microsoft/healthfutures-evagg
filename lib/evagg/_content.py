import json
from typing import Any, Dict, List, Sequence

from lib.evagg.lit import IFindVariantMentions
from lib.evagg.sk import ISemanticKernelClient
from lib.evagg.types import IPaperQuery, Paper

from ._interfaces import IExtractFields


class SimpleContentExtractor(IExtractFields):
    def __init__(self, fields: Sequence[str]) -> None:
        self._fields = fields

    def _field_to_value(self, field: str) -> str:
        if field == "gene":
            return "CHI3L1"
        if field == "variant":
            return "p.Y34C"
        if field == "MOI":
            return "AD"
        if field == "phenotype":
            return "Long face (HP:0000276)"
        if field == "functional data":
            return "No"
        else:
            return "Unknown"

    def extract(self, paper: Paper, query: IPaperQuery) -> Sequence[dict[str, str]]:
        # Dummy implementation that returns a single variant with a static set of fields.
        return [{field: self._field_to_value(field) for field in self._fields}]


class SemanticKernelContentExtractor(IExtractFields):
    _SUPPORTED_FIELDS = {"gene", "paper_id", "hgvsc", "hgvsp", "phenotype", "zygosity", "inheritance"}

    def __init__(
        self, fields: Sequence[str], sk_client: ISemanticKernelClient, mention_finder: IFindVariantMentions
    ) -> None:
        self._fields = fields
        self._sk_client = sk_client
        self._mention_finder = mention_finder

    def _excerpt_from_mentions(self, mentions: Sequence[dict[str, Any]]) -> str:
        return "\n\n".join([m["text"] for m in mentions])

    def extract(self, paper: Paper, query: IPaperQuery) -> Sequence[dict[str, str]]:
        # Find all the variant mentions in the paper relating to the query.
        variant_mentions = self._mention_finder.find_mentions(query, paper)

        # For each variant/field pair, extract the appropriate content.
        results: List[dict[str, str]] = []

        for variant_id in variant_mentions.keys():
            mentions = variant_mentions[variant_id]
            variant_results: dict[str, str] = {"variant": variant_id}

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


class TruthsetContentExtractor(IExtractFields):
    def __init__(self, field_map: Sequence[Dict[str, str]]) -> None:
        # Turn list of single-element key-mapping dicts into a list of tuples.
        self._field_map = [kv for [kv] in [kv.items() for kv in field_map]]

    def extract(self, paper: Paper, query: IPaperQuery) -> Sequence[dict[str, str]]:
        extracted_fields = []
        # For each queried variant for which we have evidence in the paper...
        for v in query.terms() & paper.evidence.keys():
            # Union the evidence props with the paper props.
            properties = paper.props | paper.evidence[v]
            # Add a new evidence dict to the list, mapping evidence values to new key names.
            extracted_fields.append({out_key: properties[k] for k, out_key in self._field_map})
        return extracted_fields
