import asyncio
import logging
from typing import Any, Dict, Sequence

from lib.evagg.llm import IPromptClient

from .interfaces import IFindVariants
from .observation import _get_prompt_file_path, run_json_prompt

logger = logging.getLogger(__name__)


class LLMVariantFinder(IFindVariants):
    _SYSTEM_PROMPT = """
You are an intelligent assistant to a genetic analyst. Their task is to identify the genetic variant or variants that
are causing a patient's disease. One approach they use to solve this problem is to seek out evidence from the academic
literature that supports (or refutes) the potential causal role that a given variant is playing in a patient's disease.

As part of that process, you will assist the analyst in identifying observations of genetic variation in human
subjects/patients.

All of your responses should be provided in the form of a JSON object. These responses should never include long,
uninterrupted sequences of whitespace characters.
"""

    def __init__(self, llm_client: IPromptClient) -> None:
        self._llm_client = llm_client

    async def _run_json_prompt(
        self, prompt_filepath: str, params: Dict[str, str], prompt_settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        return await run_json_prompt(self._llm_client, prompt_filepath, params, prompt_settings, self._SYSTEM_PROMPT)

    async def find_variant_descriptions(
        self, full_text: str, focus_texts: Sequence[str] | None, gene_symbol: str, metadata: Dict[str, Any]
    ) -> Sequence[str]:
        """Identify the genetic variants relevant to the gene_symbol described in the full text of the paper.

        Returned variants will be _as described_ in the source text. Downstream manipulations to make them
        HGVS-compliant may be required.
        """
        # Create prompts to find all the unique variants mentioned in the full text and focus texts.
        prompt_runs = [
            self._run_json_prompt(
                prompt_filepath=_get_prompt_file_path("find_variants"),
                params={"text": text, "gene_symbol": gene_symbol},
                prompt_settings={"prompt_tag": "observation__find_variants", "prompt_metadata": metadata},
            )
            for text in ([full_text] if full_text else []) + list(focus_texts or [])
        ]

        # Run prompts in parallel.
        responses = await asyncio.gather(*prompt_runs)

        # Often, the gene-symbol is provided as a prefix to the variant, remove it.
        # Note: we do additional similar checks later, but it's useful to do it now to reduce redundancy.
        def _strip_gene_symbol(x: str) -> str:
            # If x starts with gene_symbol, remove that and any subsequent colon.
            if x.startswith(gene_symbol):
                return x[len(gene_symbol) :].lstrip(":")
            return x

        candidates = list({_strip_gene_symbol(v) for r in responses for v in r.get("variants", []) if v != "unknown"})

        # Seems like this should be unnecessary, but remove the example variants from the list of candidates.
        example_variant_subs = [
            "1234A>T",
            "2345del",
            "K34T",
            "rs123456789",
            "A55T",
            "Ala55Lys",
            "K412T",
            "Ser195Gly",
            "T65I",
            "1234G>T",
            "4321A>G",
            "1234567A>T",
        ]

        if any(ex in ca for ex in example_variant_subs for ca in candidates):
            candidates_from_examples = [ca for ca in candidates if any(ex in ca for ex in example_variant_subs)]
            logger.warning(
                f"Removing example variants found in candidates for {gene_symbol}: {candidates_from_examples}"
            )
            candidates = [ca for ca in candidates if ca not in candidates_from_examples]

        # If the variant is reported with both coding and protein-level
        # descriptions, split these into two with another prompt.
        split_prompt_runs = []
        for i in reversed(range(len(candidates))):
            if "p." in candidates[i] and "c." in candidates[i]:
                split_prompt_runs.append(
                    self._run_json_prompt(
                        prompt_filepath=_get_prompt_file_path("split_variants"),
                        params={"variant_list": f'"{candidates[i]}"'},  # Encase in double-quotes for bulk calling.
                        prompt_settings={"prompt_tag": "observation__split_variants", "prompt_metadata": metadata},
                    )
                )
                del candidates[i]
        # Run split prompts in parallel.
        split_responses = await asyncio.gather(*split_prompt_runs)
        # Add the split variants back in to the candidates list.
        candidates.extend(v for r in split_responses for v in r.get("variants", []))

        return candidates