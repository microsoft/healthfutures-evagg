import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Sequence, Tuple

from lib.evagg.llm import IPromptClient
from lib.evagg.ref import IFetchHPO, ISearchHPO
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
        "phenotype": os.path.dirname(__file__) + "/prompts/phenotypes_all.txt",
        "variant_type": os.path.dirname(__file__) + "/prompts/variant_type.txt",
        "engineered_cells": os.path.dirname(__file__) + "/prompts/functional_study.txt",
        "patient_cells_tissues": os.path.dirname(__file__) + "/prompts/functional_study.txt",
        "animal_model": os.path.dirname(__file__) + "/prompts/functional_study.txt",
    }
    # These are the expensive prompt fields we should cache per paper.
    _CACHE_VARIANT_FIELDS = ["variant_type", "functional_study"]
    _CACHE_INDIVIDUAL_FIELDS = ["phenotype"]

    # Read the system prompt from file
    _SYSTEM_PROMPT = open(os.path.dirname(__file__) + "/prompts/system.txt").read()

    _DEFAULT_PROMPT_SETTINGS = {
        "max_tokens": 2048,
        "prompt_tag": "observation",
        "temperature": 0.7,
        "top_p": 0.95,
        "response_format": {"type": "json_object"},
    }

    def __init__(
        self,
        fields: Sequence[str],
        llm_client: IPromptClient,
        observation_finder: IFindObservations,
        phenotype_searcher: ISearchHPO,
        phenotype_fetcher: IFetchHPO,
        prompt_settings: Dict[str, Any] | None = None,
    ) -> None:
        # Pre-check the list of fields being requested.
        supported_fields = self._STATIC_FIELDS + self._PAPER_FIELDS + list(self._PROMPT_FIELDS)
        if any(f not in supported_fields for f in fields):
            raise ValueError()

        self._fields = fields
        self._llm_client = llm_client
        self._observation_finder = observation_finder
        self._phenotype_searcher = phenotype_searcher
        self._phenotype_fetcher = phenotype_fetcher
        self._instance_prompt_settings = (
            {**self._DEFAULT_PROMPT_SETTINGS, **prompt_settings} if prompt_settings else self._DEFAULT_PROMPT_SETTINGS
        )

    def _excerpt_from_mentions(self, mentions: Sequence[Dict[str, Any]]) -> str:
        return "\n\n".join([m["text"] for m in mentions])

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

        prompt_settings = {**self._instance_prompt_settings, **prompt_settings}

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
        """Convert a list of unstructured phenotype descriptions to HPO/OMIM terms."""
        if not phenotype:
            return []

        match_dict = {}

        # Any descriptions that look like valid HPO terms themselves should be validated.
        for term in phenotype.copy():
            ids = re.findall(r"\(?HP:\d+\)?", term)
            if ids:
                hpo_id = ids[0]
                id_result = self._phenotype_fetcher.fetch(hpo_id.strip("()"))
                if id_result:
                    phenotype.remove(term)
                    match_dict[term] = f"{id_result['name']} ({id_result['id']})"

        # Search for each term in its entirety, for those that we match, save them and move on.
        for term in phenotype.copy():
            result = self._phenotype_searcher.search(query=term.split("(")[0].strip(), retmax=1)
            if result:
                phenotype.remove(term)
                match_dict[term] = f"{result[0]['name']} ({result[0]['id']})"

        # For those we don't match, search for each word within the term and collect the unique results.
        for term in phenotype.copy():
            words = term.split()
            candidates = set()
            for word in words:
                if word.lower() == "unknown":
                    continue
                retmax = 1
                result = self._phenotype_searcher.search(query=word, retmax=retmax)
                if result:
                    for i in range(min(retmax, len(result))):
                        candidates.add(f"{result[i]['name']} ({result[i]['id']})")

            # Ask AOAI to determine which of the unique results is the best match.
            if candidates:
                params = {"term": term, "candidates": ", ".join(candidates)}
                response = await self._run_json_prompt(
                    os.path.dirname(__file__) + "/prompts/phenotypes_candidates.txt",
                    params,
                    {
                        "prompt_tag": "phenotypes_candidates",
                    },
                )
                if match := response.get("match"):
                    match_dict[term] = match
                    phenotype.remove(term)

        logger.debug(f"Converted phenotypes: {match_dict}")

        all_values = list(match_dict.values())
        all_values.extend(phenotype)
        return all_values

    async def _observation_phenotypes_for_text(
        self, text: str, observation_description: str, gene_symbol: str
    ) -> List[str]:

        all_phenotypes_result = await self._run_json_prompt(
            self._PROMPT_FIELDS["phenotype"],
            {"passage": text},
            {"prompt_tag": "phenotypes_all", "max_tokens": 4096},
        )
        if (all_phenotypes := all_phenotypes_result.get("phenotypes", [])) == []:
            return []

        # TODO: consider linked observations like comp-hets?
        observation_phenotypes_params = {
            "gene": gene_symbol,
            "passage": text,
            "observation": observation_description,
            "candidates": ", ".join(all_phenotypes),
        }
        observation_phenotypes_result = await self._run_json_prompt(
            os.path.dirname(__file__) + "/prompts/phenotypes_observation.txt",
            observation_phenotypes_params,
            {"prompt_tag": "phenotypes_observation"},
        )
        if (observation_phenotypes := observation_phenotypes_result.get("phenotypes", [])) == []:
            return []

        observation_acronymns_result = await self._run_json_prompt(
            os.path.dirname(__file__) + "/prompts/phenotypes_acronyms.txt",
            {"passage": text, "phenotypes": ", ".join(observation_phenotypes)},
            {"prompt_tag": "phenotypes_acronyms"},
        )

        return observation_acronymns_result.get("phenotypes", [])

    async def _generate_phenotype_field(self, gene_symbol: str, observation: Observation) -> str:
        # Obtain all the phenotype strings listed in the text associated with the gene.
        fulltext = "\n\n".join([t.text for t in observation.texts])
        # TODO: treating all tables in paper as a single text, maybe this isn't ideal, consider grouping by 'id'
        table_texts = "\n\n".join([t.text for t in observation.texts if t.section_type == "TABLE"])

        # Determine the phenotype strings that are associated specifically with the observation.
        v_sub = ", ".join(observation.variant_descriptions)
        if observation.patient_descriptions != ["unknown"]:
            p_sub = ", ".join(observation.patient_descriptions)
            obs_desc = f"the patient described as {p_sub} who possesses the variant described as {v_sub}."
        else:
            obs_desc = f"the variant described as {v_sub}."

        # Run phenotype extraction for all the texts of interest.
        texts = [fulltext]
        if table_texts != "":
            texts.append(table_texts)
        result = await asyncio.gather(*[self._observation_phenotypes_for_text(t, obs_desc, gene_symbol) for t in texts])
        observation_phenotypes = list({item for sublist in result for item in sublist})

        # Now convert this phenotype list to OMIM/HPO ids.
        structured_phenotypes = await self._convert_phenotype_to_hpo(observation_phenotypes)

        # Duplicates are conceivable, get unique set again.
        return ", ".join(set(structured_phenotypes))

    async def _run_field_prompt(self, gene_symbol: str, observation: Observation, field: str) -> Dict[str, Any]:
        params = {
            # First element is full text of the observation, consider alternatives
            "passage": "\n\n".join([t.text for t in observation.texts]),
            "variant_descriptions": ", ".join(observation.variant_descriptions),
            "patient_descriptions": ", ".join(observation.patient_descriptions),
            "gene": gene_symbol,
        }
        return await self._run_json_prompt(self._PROMPT_FIELDS[field], params, {"prompt_tag": field})

    async def _generate_basic_field(self, gene_symbol: str, observation: Observation, field: str) -> str:
        result = (await self._run_field_prompt(gene_symbol, observation, field)).get(field, "failed")
        # result can be a string or a json object.
        # TODO: A little wonky that we're forcing a string here, when really we should be more permissive.
        if not isinstance(result, str):
            result = json.dumps(result)
        return result

    async def _generate_functional_study_field(self, gene_symbol: str, observation: Observation, field: str) -> str:
        result = await self._run_field_prompt(gene_symbol, observation, field)
        func_studies = result.get("functional_study", [])

        # Note the prompt uses a different set of strings to represent the study types found, so we need to map them.
        map = {
            "engineered_cells": "cell line",
            "patient_cells_tissues": "patient cells",
            "animal_model": "animal model",
            "none": "none",
        }

        return "True" if (map[field] in func_studies) else "False"

    async def _generate_prompt_field(self, gene_symbol: str, observation: Observation, field: str) -> str:
        if field == "phenotype":
            return await self._generate_phenotype_field(gene_symbol, observation)
        elif field in ["engineered_cells", "patient_cells_tissues", "animal_model"]:
            return await self._generate_functional_study_field(gene_symbol, observation, field)
        else:
            return await self._generate_basic_field(gene_symbol, observation, field)

    def _get_fixed_field(self, gene_symbol: str, ob: Observation, field: str) -> Tuple[str, str]:
        if field == "gene":
            value = gene_symbol
        elif field == "individual_id":
            value = ob.individual
        elif field == "hgvs_c":
            value = ob.variant.hgvs_desc if not ob.variant.hgvs_desc.startswith("p.") else "NA"
        elif field == "hgvs_p":
            if ob.variant.protein_consequence:
                value = ob.variant.protein_consequence.hgvs_desc
            else:
                value = ob.variant.hgvs_desc if ob.variant.hgvs_desc.startswith("p.") else "NA"
        elif field == "transcript":
            value = ob.variant.refseq if ob.variant.refseq else "unknown"
        elif field == "valid":
            value = str(ob.variant.valid)
        elif field == "gnomad_frequency":
            value = "unknown"
        else:
            raise ValueError(f"Unsupported field: {field}")
        return field, value

    async def _get_fields(
        self,
        gene_symbol: str,
        ob: Observation,
        fields: Dict[str, str],
        cache: Dict[Tuple[HGVSVariant | str, str], asyncio.Task],
    ) -> Dict[str, str]:

        async def _get_prompt_field(field: str) -> Tuple[str, str]:
            # Use a cached task for variant fields if available.
            if (ob.variant, field) in cache:
                prompt_task = cache[(ob.variant, field)]
                logger.info(f"Using cached task for {ob.variant} {field}")
            elif (ob.individual, field) in cache:
                prompt_task = cache[(ob.individual, field)]
                logger.info(f"Using cached task for {ob.individual} {field}")
            else:
                # Create and schedule a prompt task to get the prompt field.
                prompt_task = asyncio.create_task(self._generate_prompt_field(gene_symbol, ob, field))
            # See if we should cache it for next time.
            if field in self._CACHE_VARIANT_FIELDS:
                cache[(ob.variant, field)] = prompt_task
            elif ob.individual != "unknown" and field in self._CACHE_INDIVIDUAL_FIELDS:
                cache[(ob.individual, field)] = prompt_task
            # Get the value from the completed task.
            value = await prompt_task
            return field, value

        # Collect any prompt-based fields and add them to the existing fields dictionary.
        fields.update(await asyncio.gather(*[_get_prompt_field(f) for f in self._PROMPT_FIELDS if f in self._fields]))
        # Collect the remaining field values and add them to the existing fields dictionary.
        fields.update(self._get_fixed_field(gene_symbol, ob, f) for f in self._fields if f not in fields)
        return fields

    async def _extract_fields(self, paper: Paper, gene_symbol: str, obs: Sequence[Observation]) -> List[Dict[str, str]]:
        # TODO - because the returned observations include the text associated with each observation, it's not trivial
        # to pre-cache the variant level fields. We don't have any easy way to collect all the unique texts associated
        # with all observations of the same variant (but different individuals). As a temporary solution, we'll cache
        # the first finding of a variant-level result and use that only. This will not be robust to scenarios where the
        # texts associated with multiple observations of the same variant differ.
        cache: Dict[Tuple[HGVSVariant | str, str], asyncio.Task] = {}
        # Precompute paper-level fields to include in each set of observation fields.
        fields = self._get_all_paper_fields(paper, [f for f in self._fields if f in self._PAPER_FIELDS])
        return await asyncio.gather(*[self._get_fields(gene_symbol, ob, fields.copy(), cache) for ob in obs])

    def extract(self, paper: Paper, gene_symbol: str) -> Sequence[Dict[str, str]]:
        if not paper.props.get("can_access", False):
            logger.warning(f"Skipping {paper.id} because it is not licensed for access")
            return []

        # Find all the observations in the paper relating to the query.
        observations = asyncio.run(self._observation_finder.find_observations(gene_symbol, paper))
        if not observations:
            logger.warning(f"No observations found in {paper.id}")
            return []

        # Extract all the requested fields from the observations.
        logger.info(f"Found {len(observations)} observations in {paper.id}")
        return asyncio.run(self._extract_fields(paper, gene_symbol, observations))
