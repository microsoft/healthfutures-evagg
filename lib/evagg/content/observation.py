import json
import logging
import os
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from lib.evagg.llm import IPromptClient, OpenAIClient
from lib.evagg.ref import IPaperLookupClient, MutalyzerClient, NcbiLookupClient, NcbiReferenceLookupClient
from lib.evagg.svc import CosmosCachingWebClient, get_dotenv_settings
from lib.evagg.types import HGVSVariant, ICreateVariants, Paper

from .mention import HGVSVariantFactory

logger = logging.getLogger(__name__)


def make_default_observation_finder() -> "ObservationFinder":

    # General web client
    web_client = CosmosCachingWebClient(cache_settings=get_dotenv_settings(filter_prefix="EVAGG_CONTENT_CACHE_"))
    # Mutalyzer needs its own web client because of the 422 no raise.
    mutalyzer_client = MutalyzerClient(
        web_client=CosmosCachingWebClient(
            web_settings={"no_raise_codes": [422]},
            cache_settings=get_dotenv_settings(filter_prefix="EVAGG_CONTENT_CACHE_"),
        )
    )
    return ObservationFinder(
        llm_client=OpenAIClient(get_dotenv_settings(filter_prefix="AZURE_OPENAI_")),
        paper_lookup_client=NcbiLookupClient(
            web_client=web_client,
            settings=get_dotenv_settings(filter_prefix="NCBI_EUTILS_"),
        ),
        variant_factory=HGVSVariantFactory(
            normalizer=mutalyzer_client,
            back_translator=mutalyzer_client,
            refseq_client=NcbiReferenceLookupClient(web_client=web_client),
        ),
    )


class ObservationFinder:
    _PROMPTS = {
        "find_patients": os.path.dirname(__file__) + "/prompts/observation/find_patients.txt",
        "find_variants": os.path.dirname(__file__) + "/prompts/observation/find_variants.txt",
        "link_entities": os.path.dirname(__file__) + "/prompts/observation/link_entities.txt",
    }
    _SYSTEM_PROMPT = """
You are an intelligent assistant to a genetic analyst. Their task is to identify the genetic variant or variants that
are causing a patient's disease. One approach they use to solve this problem is to seek out evidence from the academic
literature that supports (or refutes) the potential causal role that a given variant is playing in a patient's disease.

As part of that process, you will assist the analyst in identifying observations of genetic variation in human
subjects/patients.

"""

    def __init__(
        self, llm_client: IPromptClient, paper_lookup_client: IPaperLookupClient, variant_factory: ICreateVariants
    ) -> None:
        self._llm_client = llm_client
        self._paper_lookup_client = paper_lookup_client
        self._variant_factory = variant_factory

    def _call_to_json_list(self, prompt_filepath: str, params: Dict[str, str], prompt_tag: str) -> Sequence[str]:
        response = self._llm_client.prompt_file(
            user_prompt_file=prompt_filepath,
            system_prompt=self._SYSTEM_PROMPT,
            params=params,
            prompt_settings={"prompt_tag": prompt_tag},
        )

        try:
            result = json.loads(response)
        except Exception:
            logger.warning(f"Failed to parse response from LLM to {prompt_filepath}: {response}")
            return []
        return result

    def _find_patients(self, full_text: str) -> Sequence[str]:
        """Identify the individuals (human subjects) described in the full text of the paper."""
        return self._call_to_json_list(
            prompt_filepath=self._PROMPTS["find_patients"],
            params={"text": full_text},
            prompt_tag="observation__find_patients",
        )

    def _find_variants(self, full_text: str, query: str) -> Sequence[str]:
        """Identify the genetic variants relevant to the query described in the full text of the paper.

        `query` should be a gene symbol.
        """
        return self._call_to_json_list(
            prompt_filepath=self._PROMPTS["find_variants"],
            params={"text": full_text, "gene_symbol": query},
            prompt_tag="observation__find_variants",
        )

    def _link_entities(
        self, full_text: str, patients: Sequence[str], variants: Sequence[str], query: str
    ) -> Dict[str, List[str]]:
        params = {
            "text": full_text,
            "patients": ", ".join(patients),
            "variants": ", ".join(variants),
            "gene_symbol": query,
        }
        response = self._llm_client.prompt_file(
            user_prompt_file=self._PROMPTS["link_entities"],
            system_prompt=self._SYSTEM_PROMPT,
            params=params,
            prompt_settings={"prompt_tag": "observation__link_entities"},
        )

        try:
            result = json.loads(response)
        except Exception:
            logger.warning(f"Failed to parse response from LLM to {self._PROMPTS['link_entities']}: {response}")
            return {}
        return result

    def _create_variant(self, variant_str: str, gene_symbol: str) -> HGVSVariant | None:
        """Create an HGVSVariant object from `variant_str` and `gene_symbol`."""
        if variant_str.find(":") >= 0:
            refseq, text_desc = variant_str.split(":", 1)
            refseq = refseq.strip()
            text_desc = text_desc.strip()
        else:
            refseq = None
            text_desc = variant_str.strip()

        try:
            return self._variant_factory.parse(text_desc, gene_symbol, refseq)
        except ValueError:
            logger.warning(f"Failed to create variant from {variant_str} and {gene_symbol}")
            return None

    def find_observations(self, query: str, paper: Paper) -> Mapping[Tuple[HGVSVariant, str], Sequence[str]]:
        """Identify all observations relevant to `query` in `paper`.

        `query` should be a gene_symbol. `paper` is the paper to search for relevant observations. Paper must be in the
        PMC-OA dataset and have license terms that permit derivative works based on current restrictions.

        Observations are logically "clinical" observations of a variant in a human, thus this function returns a dict
        keyed by tuples of variants and string representations of the individual in which that variant was observed. The
        values in this dictionary are a collection of mentions relevant to this observation throughout the paper.
        """
        # Obtain the full-text of the paper.
        full_text = self._paper_lookup_client.full_text(paper)
        if full_text is None:
            logger.warning(f"Skipping {paper.id} because full text could not be retrieved")
            return {}

        # Determine all of the patients specifically referred to in the paper, if any.
        # TODO: handle deduping patients
        patients = self._find_patients(full_text)
        logger.debug(f"Found the following patients in {paper}: {patients}")
        if patients == ["unknown"]:
            patients = []

        # Determine all of the genetic variants matching `query`
        # TODO: handle deduping variants
        variants = self._find_variants(full_text, query)
        logger.debug(f"Found the following variants for {query} in {paper}: {variants}")
        if variants == ["unknown"]:
            variants = []

        # Obtain a mapping of valid variant strings to the corresponding variant object.
        variant_objects = {v: self._create_variant(v, query) for v in variants}
        variant_objects = {k: v for k, v in variant_objects.items() if v is not None and v.valid}
        validated_variants = list(variant_objects.keys())

        # If there are both variants and patients, build a mapping between the two,
        # if there are only variants and no patients, no need to link, just assign all the variants to "unknown".
        # if there are no variants (regardless of patients), then there are no observations to report.
        if variants and patients:
            observations = self._link_entities(full_text, patients, validated_variants, query)
        elif variants:
            observations = {"unknown": validated_variants}
        else:
            observations = {}

        # TODO: don't just use the full text of the paper here.
        result = {}
        for individual, variant_strs in observations.items():
            for variant_str in variant_strs:
                variant = variant_objects.get(variant_str)
                if variant is None:
                    logger.warning(f"Variant {variant_str} not found in variant_objects")
                    continue
                # TODO: As above, this should instead go get tagged sections of text for token efficiency.
                result[variant, individual] = [full_text]

        return result
