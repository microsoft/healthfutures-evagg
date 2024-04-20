import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Sequence, Tuple

from lib.evagg.llm import IPromptClient
from lib.evagg.ref import ISearchHPO
from lib.evagg.types import HGVSVariant, Paper

from ..interfaces import IExtractFields
from .interfaces import IFindObservations, Observation

logger = logging.getLogger(__name__)


class PromptBasedContentExtractor(IExtractFields):
    _PAPER_FIELDS = ["paper_id", "study_type", "citation"]
    _STATIC_FIELDS = ["gene", "hgvs_c", "hgvs_p", "transcript", "gnomad_frequency", "valid", "individual_id"]
    _PROMPT_FIELDS = {
        "zygosity": os.path.dirname(__file__) + "/prompts/zygosity.txt",
        "variant_inheritance": os.path.dirname(__file__) + "/prompts/variant_inheritance.txt",
        "phenotype": os.path.dirname(__file__) + "/prompts/phenotype.txt",
        "phenotype_to_hpo": os.path.dirname(__file__) + "/prompts/phenotype_to_hpo.txt",
        "variant_type": os.path.dirname(__file__) + "/prompts/variant_type.txt",
        "functional_study": os.path.dirname(__file__) + "/prompts/functional_study.txt",
    }
    # These are expensive prompt fields
    _CACHE_FIELDS = ["variant_type", "functional_study"]
    _SYSTEM_PROMPT = """
You are an intelligent assistant to a genetic analyst. Their task is to identify the genetic variant or variants that
are causing a patient's disease. One approach they use to solve this problem is to seek out evidence from the academic
literature that supports (or refutes) the potential causal role that a given variant is playing in a patient's disease.

As part of that process, you will assist the analyst in collecting specific details about genetic variants that have
been observed in the literature.

All of your responses should be provided in the form of a JSON object. These responses should never include long,
uninterrupted sequences of whitespace characters.
"""

    def __init__(
        self,
        fields: Sequence[str],
        llm_client: IPromptClient,
        observation_finder: IFindObservations,
        phenotype_searcher: ISearchHPO,
    ) -> None:
        # Pre-check the list of fields being requested.
        supported_fields = self._STATIC_FIELDS + self._PAPER_FIELDS + list(self._PROMPT_FIELDS)
        if any(f not in supported_fields for f in fields):
            raise ValueError()

        self._fields = fields
        self._llm_client = llm_client
        self._observation_finder = observation_finder
        self._phenotype_searcher = phenotype_searcher

    def _excerpt_from_mentions(self, mentions: Sequence[Dict[str, Any]]) -> str:
        return "\n\n".join([m["text"] for m in mentions])

    def _can_process_paper(self, paper: Paper) -> bool:
        if (
            "pmcid" not in paper.props
            or paper.props["pmcid"] == ""
            or "is_pmc_oa" not in paper.props
            or paper.props["is_pmc_oa"] is False
        ):
            return False
        return True

    def _get_all_paper_fields(self, paper: Paper, fields: Sequence[str]) -> Dict[str, str]:
        result = {}
        for field in fields:
            if field == "paper_id":
                result[field] = paper.id
            elif field == "study_type":
                result[field] = paper.props.get("study_type", "TODO")
            elif field == "citation":
                result[field] = paper.citation
            else:
                raise ValueError(f"Unsupported paper field: {field}")
        return result

    async def _run_json_prompt(
        self, prompt_filepath: str, params: Dict[str, str], prompt_settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        default_settings = {
            "max_tokens": 2048,
            "prompt_tag": "observation",
            "temperature": 0.7,
            "top_p": 0.95,
            "response_format": {"type": "json_object"},
        }
        prompt_settings = {**default_settings, **prompt_settings}

        response = await self._llm_client.prompt_file(
            user_prompt_file=prompt_filepath,
            system_prompt=self._SYSTEM_PROMPT,
            params=params,
            prompt_settings=prompt_settings,
        )

        try:
            result = json.loads(response)
        except Exception:
            logger.warning(f"Failed to parse response from LLM to {prompt_filepath}: {response}")
            return {}

        return result

    async def _convert_phenotype_to_hpo(self, phenotype: List[str]) -> List[str]:
        if not isinstance(phenotype, list):
            return ["unknown"]

        response = await self._run_json_prompt(
            self._PROMPT_FIELDS["phenotype_to_hpo"],
            {"phenotypes": ", ".join(phenotype)},
            {"prompt_tag": "phenotype_to_hpo"},
        )

        # For the matched terms, validate that they're in the HPO, else drop the HPO identifier and move them to
        # unmatched.
        matched = response.get("matched", [])
        unmatched = response.get("unmatched", [])
        for m in matched.copy():
            ids = re.findall(r"\(?HP:\d+\)?", m)
            if not ids:
                logger.info(f"Unable to find HPO identifier in {m}, shifting to unmatched.")
                unmatched.append(m)
                matched.remove(m)
            else:
                if len(ids) > 1:
                    logger.info(f"Multiple HPO identifiers found in {m}. Ignoring all but the first.")
                id = ids[0]
                if not self._phenotype_searcher.exists(id.strip("()")):
                    logger.info(f"Unable to match {m} as an HPO term, searching for alternatives.")
                    unmatched.append(m.replace(id, "").strip())
                    matched.remove(m)

        # Try to use a query-based search for the unmatched terms.
        for u in unmatched:
            result = self._phenotype_searcher.search(u)
            if result:
                matched.append(f"{result['name']} ({result['id']})")
            else:
                # If we can't find a match, just use the original term.
                matched.append(u)

        return matched

    async def _get_prompt_field(self, gene_symbol: str, observation: Observation, field: str) -> str:
        params = {
            "passage": "\n\n".join(observation.texts),
            "variant_descriptions": ", ".join(observation.variant_descriptions),
            "patient_descriptions": ", ".join(observation.patient_descriptions),
            "gene": gene_symbol,
        }
        response = await self._run_json_prompt(self._PROMPT_FIELDS[field], params, {"prompt_tag": field})
        # result can be a string or a json object.
        result = response.get(field, "failed")
        if field == "phenotype":
            result = await self._convert_phenotype_to_hpo(result)  # type: ignore
        # TODO: A little wonky that we're forcing a string here, when really we should be more permissive.
        if not isinstance(result, str):
            result = json.dumps(result)
        return result

    async def _get_fields(
        self, gene_symbol: str, ob: Observation, fields: Dict[str, str], cache: dict[Tuple[HGVSVariant, str], str]
    ) -> Dict[str, str]:
        # Run through each requested field not already in the fields.
        async def _add_field(field: str) -> None:
            # Use a cached result for variant fields if available.
            if (ob.variant, field) in cache:
                fields[field] = cache[(ob.variant, field)]
            # Run a prompt to get the prompt fields.
            elif field in self._PROMPT_FIELDS:
                fields[field] = await self._get_prompt_field(gene_symbol, ob, field)
                # See if we should cache it for next time.
                if field in self._CACHE_FIELDS:
                    cache[(ob.variant, field)] = fields[field]
            elif field == "gene":
                fields[field] = gene_symbol
            elif field == "individual_id":
                fields[field] = ob.individual
            elif field == "hgvs_c":
                fields[field] = ob.variant.hgvs_desc if not ob.variant.hgvs_desc.startswith("p.") else "NA"
            elif field == "hgvs_p":
                if ob.variant.protein_consequence:
                    fields[field] = ob.variant.protein_consequence.hgvs_desc
                else:
                    fields[field] = ob.variant.hgvs_desc if ob.variant.hgvs_desc.startswith("p.") else "NA"
            elif field == "transcript":
                fields[field] = ob.variant.refseq if ob.variant.refseq else "unknown"
            elif field == "valid":
                fields[field] = str(ob.variant.valid)
            elif field == "gnomad_frequency":
                fields[field] = "unknown"
            else:
                raise ValueError(f"Unsupported field: {field}")

        await asyncio.gather(*[_add_field(field) for field in self._fields if field not in fields])
        return fields

    async def _extract_fields(self, paper: Paper, gene_symbol: str, obs: Sequence[Observation]) -> List[Dict[str, str]]:
        # TODO - because the returned observations include the text associated with each observation, it's not trivial
        # to pre-cache the variant level fields. We don't have any easy way to collect all the unique texts associated
        # with all observations of the same variant (but different individuals). As a temporary solution, we'll cache
        # the first finding of a variant-level result and use that only. This will not be robust to scenarios where the
        # texts associated with multiple observations of the same variant differ.
        cache: dict[Tuple[HGVSVariant, str], str] = {}
        # Precompute paper-level fields to include in each set of observation fields.
        fields = self._get_all_paper_fields(paper, [f for f in self._fields if f in self._PAPER_FIELDS])
        return await asyncio.gather(*[self._get_fields(gene_symbol, ob, fields.copy(), cache) for ob in obs])

    def extract(self, paper: Paper, gene_symbol: str) -> Sequence[Dict[str, str]]:
        if not self._can_process_paper(paper):
            logger.warning(f"Skipping {paper.id} because it is not in PMC-OA")
            return []

        # Find all the observations in the paper relating to the query.
        observations = asyncio.run(self._observation_finder.find_observations(gene_symbol, paper))
        if not observations:
            logger.warning(f"No observations found in {paper.id}")
            return []

        # Extract all the requested fields from the observations.
        logger.info(f"Found {len(observations)} observations in {paper.id}")
        return asyncio.run(self._extract_fields(paper, gene_symbol, observations))
