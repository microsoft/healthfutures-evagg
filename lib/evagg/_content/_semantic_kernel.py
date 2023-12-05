import json
from typing import Any, Dict, List, Sequence

from lib.evagg.lit import IFindVariantMentions
from lib.evagg.ref import INcbiSnpClient
from lib.evagg.sk import ISemanticKernelClient
from lib.evagg.types import IPaperQuery, Paper

from .._interfaces import IExtractFields


class SemanticKernelContentExtractor(IExtractFields):
    _SUPPORTED_FIELDS = {"gene", "paper_id", "hgvs_c", "hgvs_p", "phenotype", "zygosity", "variant_inheritance"}

    def __init__(
        self,
        fields: Sequence[str],
        sk_client: ISemanticKernelClient,
        mention_finder: IFindVariantMentions,
        ncbi_snp_client: INcbiSnpClient,
    ) -> None:
        self._fields = fields
        self._sk_client = sk_client
        self._mention_finder = mention_finder
        self._ncbi_snp_client = ncbi_snp_client

    def _excerpt_from_mentions(self, mentions: Sequence[Dict[str, Any]]) -> str:
        return "\n\n".join([m["text"] for m in mentions])

    def extract(self, paper: Paper, query: IPaperQuery) -> Sequence[Dict[str, str]]:
        # Only process papers in PMC.
        if "pmcid" not in paper.props or paper.props["pmcid"] == "":
            return []

        # Find all the variant mentions in the paper relating to the query.
        variant_mentions = self._mention_finder.find_mentions(query, paper)

        print(f"Found {len(variant_mentions)} variant mentions in {paper.id}")

        # Build a cached list of hgvs formats for dbsnp identifiers.
        hgvs_cache = self._ncbi_snp_client.hgvs_from_rsid([v for v in variant_mentions.keys() if v.startswith("rs")])

        # For each variant/field pair, extract the appropriate content.
        results: List[Dict[str, str]] = []

        for variant_id in variant_mentions.keys():
            mentions = variant_mentions[variant_id]
            variant_results: Dict[str, str] = {"variant": variant_id}

            # Simplest thing we can think of is to just concatenate all the chunks.
            paper_excerpts = self._excerpt_from_mentions(mentions)
            gene_symbol = mentions[0].get("gene_symbol", "unknown")  # Mentions should never be empty.

            # If we have a cached hgvs value, use it.
            hgvs: Dict[str, str] = {}
            if variant_id in hgvs_cache:
                hgvs = hgvs_cache[variant_id]
            elif variant_id.startswith("c."):
                hgvs = {"hgvs_c": variant_id}
            elif variant_id.startswith("p."):
                hgvs = {"hgvs_p": variant_id}

            for field in self._fields:
                if field not in self._SUPPORTED_FIELDS:
                    raise ValueError(f"Unsupported field: {field}")

                if field == "gene":
                    result = gene_symbol
                elif field == "paper_id":
                    result = paper.id
                elif field == "hgvs_c":
                    result = hgvs["hgvs_c"] if ("hgvs_c" in hgvs and hgvs["hgvs_c"]) else "unknown"
                elif field == "hgvs_p":
                    result = hgvs["hgvs_p"] if ("hgvs_p" in hgvs and hgvs["hgvs_p"]) else "unknown"
                else:
                    context_variables = {"input": paper_excerpts, "variant": variant_id, "gene": gene_symbol}

                    raw = self._sk_client.run_completion_function(
                        skill="content", function=field, context_variables=context_variables
                    )
                    try:
                        result = json.loads(raw)[field]
                    except Exception:
                        result = "failed"
                variant_results[field] = result
            results.append(variant_results)
        return results
