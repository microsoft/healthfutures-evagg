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
        "phenotype": os.path.dirname(__file__) + "/prompts/phenotype.txt",
        "variant_type": os.path.dirname(__file__) + "/prompts/variant_type.txt",
        "functional_study": os.path.dirname(__file__) + "/prompts/functional_study.txt",
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
        if not isinstance(phenotype, list) or phenotype == ["unknown"] or phenotype == ["Unknown"]:
            return ["unknown"]

        # Give AOAI a chance to come up with the phenotype terms on its own.
        response = await self._run_json_prompt(
            os.path.dirname(__file__) + "/prompts/phenotype_to_hpo.txt",
            {"phenotypes": ", ".join(phenotype)},
            {"prompt_tag": "phenotype_to_hpo"},
        )

        # Validate AOAI's responses.
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
                move = False
                # Obtain the HPO term based on the HPO id.
                if not bool(id_result := self._phenotype_fetcher.fetch(id.strip("()"))):
                    logger.info(f"Term {m} contains invalid HPO id.")
                    move = True
                # Obtain the HPO term based on the string description.
                if not bool(desc_result := self._phenotype_fetcher.fetch(m.split("(")[0].strip())):
                    logger.info(f"Term {m} contains invalid HPO description.")
                    move = True
                # Only keep this term if both exist and they match.
                if not id_result or not desc_result or id_result != desc_result:
                    logger.info(f"Term {m} has mismatched HPO description {desc_result} and HPO id {id_result}.")
                    move = True
                if move:
                    # Strip the HPO identifier and move it to unmatched.
                    unmatched.append(m.split("(")[0].strip())
                    matched.remove(m)

        # For anything that remains unmatched, use an HPO search client to try to find the closest match.
        # TODO: alternatively, take a broader approach here where we search on each word within the unmatched term,
        # gather the results, and then ask AOAI (via another prompt) to select from all those terms which seems best.
        # Try to use a query-based search for the unmatched terms.
        for u in unmatched:
            if u.lower() == "unknown":
                logger.warning("Unknown value made it into phenotype list.")
                continue
            result = self._phenotype_searcher.search(query=u, retmax=1)
            if result:
                logger.info(f"Rescued term {u} with HPO term {result[0]['name']} ({result[0]['id']}).")
                matched.append(f"{result[0]['name']} ({result[0]['id']})")
            else:
                logger.info(f"Failed to rescue term {u}.")
                matched.append(u)

        if any("ialeptic" in m for m in matched):
            import pdb
            pdb.set_trace()
            
        return matched

    async def _generate_prompt_field(self, gene_symbol: str, observation: Observation, field: str) -> str:
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
            elif field in self._CACHE_INDIVIDUAL_FIELDS:
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
