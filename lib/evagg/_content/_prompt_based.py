import json
import os
from typing import Any, Dict, List, Sequence

from lib.evagg.lit import IFindVariantMentions
from lib.evagg.llm.openai import IOpenAIClient
from lib.evagg.ref import INcbiSnpClient
from lib.evagg.types import IPaperQuery, Paper

from .._interfaces import IExtractFields


class PromptBasedContentExtractor(IExtractFields):
    _SUPPORTED_FIELDS = {"gene", "paper_id", "hgvs_c", "hgvs_p", "phenotype", "zygosity", "variant_inheritance"}
    _PROMPTS = {
        "zygosity": os.path.dirname(__file__) + "/prompts/zygosity.txt",
        "variant_inheritance": os.path.dirname(__file__) + "/prompts/variant_inheritance.txt",
        "phenotype": os.path.dirname(__file__) + "/prompts/phenotype.txt",
    }

    def __init__(
        self,
        fields: Sequence[str],
        llm_client: IOpenAIClient,
        mention_finder: IFindVariantMentions,
        ncbi_snp_client: INcbiSnpClient,
    ) -> None:
        self._fields = fields
        self._llm_client = llm_client
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

        # TODO, variant_id can currently be any of the following:
        # - rsid (e.g., rs123456789)
        # - hgvs_c (e.g., c.123A>T)
        # - hgvs_p (e.g., p.Ala123Thr)
        # - gene+hgvs (e.g., BRCA1:c.123A>T || BRCA1:p.Ala123Thr)
        #
        # Currently handling this below in a hacky way temporarily. Need to figure out
        # the correct story for variant nomenclature.

        for variant_id in variant_mentions.keys():
            mentions = variant_mentions[variant_id]
            variant_results: Dict[str, str] = {"variant": variant_id}

            print(f"### Extracting fields for {variant_id} in {paper.id}")

            # Simplest thing we can think of is to just concatenate all the chunks.
            paper_excerpts = self._excerpt_from_mentions(mentions)
            gene_symbol = mentions[0].get("gene_symbol", "unknown")  # Mentions should never be empty.

            # If we have a cached hgvs value, use it. This means variant_id is an rsid.
            hgvs: Dict[str, str] = {}
            if variant_id in hgvs_cache:
                hgvs = hgvs_cache[variant_id]
            elif variant_id.startswith("c."):
                hgvs = {"hgvs_c": variant_id}
            elif variant_id.startswith("p."):
                hgvs = {"hgvs_p": variant_id}
            else:  # assume variant_id is gene+hgvs
                hgvs_unk = variant_id.split(":")
                if len(hgvs_unk) == 2:
                    if hgvs_unk[1].startswith("c."):
                        hgvs = {"hgvs_c": hgvs_unk[1]}
                    elif hgvs_unk[1].startswith("p."):
                        hgvs = {"hgvs_p": hgvs_unk[1]}

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
                    params = {"passage": paper_excerpts, "variant": variant_id, "gene": gene_symbol}

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
                variant_results[field] = result
            results.append(variant_results)
        return results
