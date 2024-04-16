import json
import logging
import os
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from lib.evagg.llm import IPromptClient
from lib.evagg.ref import ISearchHPO
from lib.evagg.types import HGVSVariant, Paper

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
        "variant_type",
        "functional_study",
        "gnomad_frequency",
        "study_type",
        "citation",
    }
    _PAPER_FIELDS = {"paper_id", "study_type", "citation"}
    _VARIANT_FIELDS = {
        "hgvs_c",
        "hgvs_p",
        "transcript",
        "valid",
        "variant_type",
        "functional_study",
        "gnomad_frequency",
        "gene",
    }

    _PROMPTS = {
        "zygosity": os.path.dirname(__file__) + "/prompts/zygosity.txt",
        "variant_inheritance": os.path.dirname(__file__) + "/prompts/variant_inheritance.txt",
        "phenotype": os.path.dirname(__file__) + "/prompts/phenotype.txt",
        "phenotype_to_hpo": os.path.dirname(__file__) + "/prompts/phenotype_to_hpo.txt",
        "variant_type": os.path.dirname(__file__) + "/prompts/variant_type.txt",
        "functional_study": os.path.dirname(__file__) + "/prompts/functional_study.txt",
    }
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
        if any(f not in self._SUPPORTED_FIELDS for f in fields):
            raise ValueError()

        self._fields = fields
        self._llm_client = llm_client
        self._observation_finder = observation_finder
        self._phenotype_searcher = phenotype_searcher

    def _excerpt_from_mentions(self, mentions: Sequence[Dict[str, Any]]) -> str:
        return "\n\n".join([m["text"] for m in mentions])

    def _can_process_paper(self, paper) -> bool:
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

    def _run_json_prompt(
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

        response = self._llm_client.prompt_file(
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

    def _convert_phenotype_to_hpo(self, phenotype: List[str]) -> List[str]:
        if not isinstance(phenotype, list):
            return ["unknown"]

        params = {"phenotypes": ", ".join(phenotype)}
        response = self._run_json_prompt(self._PROMPTS["phenotype_to_hpo"], params, {"prompt_tag": "phenotype_to_hpo"})

        # For the matched terms, validate that they're in the HPO, else drop the HPO identifier and move them to
        # unmatched.
        matched = response.get("matched", [])
        unmatched = response.get("unmatched", [])
        for m in matched.copy():
            id = re.findall(r"\(?HP:\d+\)?", m)
            if not id:
                logger.info(f"Unable to find HPO identifier in {m}, shifting to unmatched.")
                unmatched.append(m)
                matched.remove(m)
            if len(id) > 1:
                logger.info(f"Multiple HPO identifiers found in {m}. Ignoring all but the first.")
            id = id[0]
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

    def _get_observation_field(
        self, gene_symbol: str, observation: Tuple[HGVSVariant, str], observation_info: Mapping[str, Any], field: str
    ) -> str:
        variant, individual = observation
        paper_excerpts = "\n\n".join(observation_info["texts"])

        if field == "gene":
            result = gene_symbol
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
        elif field == "gnomad_frequency":
            result = "unknown"
        else:
            variant_desc = ", ".join(observation_info["variant_descriptions"])
            patient_desc = ", ".join(observation_info["patient_descriptions"])
            params = {
                "passage": paper_excerpts,
                "variant_descriptions": variant_desc,
                "patient_descriptions": patient_desc,
                "gene": gene_symbol,
            }
            response = self._run_json_prompt(self._PROMPTS[field], params, {"prompt_tag": field})
            # result can be a string or a json object.
            result = response.get(field, "failed")
            if field == "phenotype":
                result = self._convert_phenotype_to_hpo(result)
            # TODO: A little wonky that we're forcing a string here, when really we should be more permissive.
            if not isinstance(result, str):
                result = json.dumps(result)

        return result

    def extract(self, paper: Paper, gene_symbol: str) -> Sequence[Dict[str, str]]:
        if not self._can_process_paper(paper):
            logger.warning(f"Skipping {paper.id} because it is not in PMC-OA")
            return []

        # Find all the observations in the paper relating to the query.
        observations = self._observation_finder.find_observations(gene_symbol, paper)

        logger.info(f"Found {len(observations)} observations in {paper.id}")

        if not observations:
            logger.warning(f"No observations found in {paper.id}")
            return []

        # Determine paper-level fields.
        paper_fields = self._get_all_paper_fields(paper, [f for f in self._fields if f in self._PAPER_FIELDS])
        non_paper_fields = [f for f in self._fields if f not in self._PAPER_FIELDS]

        # TODO - because the returned observations include the text associated with each observation, it's not trivial
        # to pre-cache the variant level fields. We don't have any easy way to collect all the unique texts associated
        # with all observations of the same variant (but different individuals). As a temporary solution, we'll cache
        # the first finding of a variant-level result and use that only. This will not be robust to scenarios where the
        # texts associated with multiple observations of the same variant differ.
        variant_field_cache: dict[Tuple[HGVSVariant, str], str] = {}

        # For each observation, extract the appropriate content.
        results: List[Dict[str, str]] = []

        for observation, observation_info in observations.items():
            # We've pre-computed the paper fields.
            observation_results: Dict[str, str] = paper_fields.copy()

            logger.info(f"Extracting fields for {observation} in {paper.id}")

            if not observation_info or not observation_info.get("texts", None):
                logger.warning(f"No observations found for {observation} in {paper.id}")
                continue

            for field in non_paper_fields:
                if (observation[0], field) in variant_field_cache:
                    # Use the cached result for variant fields if available.
                    result = variant_field_cache[(observation[0], field)]
                else:
                    result = self._get_observation_field(gene_symbol, observation, observation_info, field)
                    # Cache the result for variant fields.
                    if field in self._VARIANT_FIELDS:
                        variant_field_cache[(observation[0], field)] = result
                observation_results[field] = result
            results.append(observation_results)
        return results
