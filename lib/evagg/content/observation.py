import asyncio
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from lib.evagg.llm import IPromptClient
from lib.evagg.ref import INormalizeVariants, IPaperLookupClient
from lib.evagg.types import HGVSVariant, ICreateVariants, Paper

from .interfaces import ICompareVariants, IFindObservations

PatientVariant = Tuple[HGVSVariant, str]

logger = logging.getLogger(__name__)


class ObservationFinderBase(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    def _get_paper_texts(self, paper: Paper) -> Dict[str, Any]:
        texts: Dict[str, Any] = {}

        texts["full_text"] = "\n".join(paper.props.get("full_text_sections", []))

        tables = {}
        root = paper.props.get("full_text_xml")
        if root is not None:
            for passage in root.findall("./passage"):
                if bool(passage.findall("infon[@key='section_type'][.='TABLE']")):
                    id = passage.find("infon[@key='id']").text
                    if not id:
                        logger.error("No id for table, using None as key")
                    if id not in tables:
                        tables[id] = passage.find("text").text
                    else:
                        tables[id] += "\n" + passage.find("text").text

        texts["tables"] = tables
        return texts


class TruthsetObservationFinder(ObservationFinderBase, IFindObservations):
    def __init__(self) -> None:
        pass

    def find_observations(self, gene_symbol: str, paper: Paper) -> Mapping[Tuple[HGVSVariant, str], Mapping[str, Any]]:
        """Identify all observations relevant to `gene_symbol` in `paper`.

        `gene_symbol` should be a gene_symbol. `paper` is the paper to search for relevant observations. Paper must be
        in the PMC-OA dataset and have license terms that permit derivative works based on current restrictions.

        Observations are logically "clinical" observations of a variant in a human, thus this function returns a dict
        keyed by tuples of variants and string representations of the individual in which that variant was observed. The
        values in this dictionary are dictionaries relevant to this observation throughout the paper. Keys in this
        dictionary include, "variant_descriptions", "patient_descriptions", "texts".
        """
        observations: Dict[Tuple[HGVSVariant, str], Mapping[str, Any]] = {}

        texts = self._get_paper_texts(paper)
        full_text = texts.get("full_text", None)

        if not full_text:
            logger.info(f"Skipping {paper.id} because full text could not be retrieved")
            return observations

        for ob in paper.evidence.keys():

            if ob[0].gene_symbol != gene_symbol:
                logger.info(f"Unexpected gene symbol found for {paper}: {ob[0].gene_symbol}")
                continue

            variant_descriptions = [ob[0].hgvs_desc]
            if paper.evidence[ob].get("paper_variant"):
                variant_descriptions.append(paper.evidence[ob]["paper_variant"])
            if ob[0].protein_consequence:
                variant_descriptions.append(ob[0].protein_consequence.hgvs_desc)

            observations[ob] = {
                "variant_descriptions": list(set(variant_descriptions)),
                "patient_descriptions": [ob[1]],
                "texts": [full_text],
            }

        return observations


class ObservationFinder(ObservationFinderBase, IFindObservations):
    _PROMPTS = {
        "check_patients": os.path.dirname(__file__) + "/prompts/observation/check_patients.txt",
        "find_genome_build": os.path.dirname(__file__) + "/prompts/observation/find_genome_build.txt",
        "find_patients": os.path.dirname(__file__) + "/prompts/observation/find_patients.txt",
        "find_variants": os.path.dirname(__file__) + "/prompts/observation/find_variants.txt",
        "link_entities": os.path.dirname(__file__) + "/prompts/observation/link_entities.txt",
        "split_patients": os.path.dirname(__file__) + "/prompts/observation/split_patients.txt",
        "split_variants": os.path.dirname(__file__) + "/prompts/observation/split_variants.txt",
    }
    _SYSTEM_PROMPT = """
You are an intelligent assistant to a genetic analyst. Their task is to identify the genetic variant or variants that
are causing a patient's disease. One approach they use to solve this problem is to seek out evidence from the academic
literature that supports (or refutes) the potential causal role that a given variant is playing in a patient's disease.

As part of that process, you will assist the analyst in identifying observations of genetic variation in human
subjects/patients.

All of your responses should be provided in the form of a JSON object. These responses should never include long,
uninterrupted sequences of whitespace characters.
"""

    def __init__(
        self,
        llm_client: IPromptClient,
        paper_lookup_client: IPaperLookupClient,
        variant_factory: ICreateVariants,
        variant_comparator: ICompareVariants,
        normalizer: INormalizeVariants,
    ) -> None:
        self._llm_client = llm_client
        self._paper_lookup_client = paper_lookup_client
        self._variant_factory = variant_factory
        self._variant_comparator = variant_comparator
        self._normalizer = normalizer

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

    async def _check_patients(self, patient_candidates: Sequence[str], texts_to_check: Sequence[str]) -> List[str]:
        checked_patients: List[str] = []

        async def check_patient(patient):
            for text in texts_to_check:
                validation_response = await self._run_json_prompt(
                    prompt_filepath=self._PROMPTS["check_patients"],
                    params={"text": text, "patient": patient},
                    prompt_settings={"prompt_tag": "observation__check_patients"},
                )
                if validation_response.get("is_patient", False) is True:
                    checked_patients.append(patient)
                    break
            if patient not in checked_patients:
                logger.debug(f"Removing {patient} from list of patients as it didn't pass final checks.")

        await asyncio.gather(*[check_patient(patient) for patient in patient_candidates])
        return checked_patients

    async def _find_patients(self, full_text: str, focus_texts: Sequence[str] | None) -> Sequence[str]:
        """Identify the individuals (human subjects) described in the full text of the paper."""
        full_text_response = await self._run_json_prompt(
            prompt_filepath=self._PROMPTS["find_patients"],
            params={"text": full_text},
            prompt_settings={"prompt_tag": "observation__find_patients"},
        )

        unique_patients = set(full_text_response.get("patients", []))

        # TODO, logically deduplicate patients here, e.g., if a patient is referred to as both "proband" and "IV-1",
        # we should ask the LLM to determine if these are the same individual.

        async def check_focus_text(focus_text):
            focus_response = await self._run_json_prompt(
                prompt_filepath=self._PROMPTS["find_patients"],
                params={"text": focus_text},
                prompt_settings={"prompt_tag": "observation__find_patients"},
            )
            unique_patients.update(focus_response.get("patients", []))

        if focus_texts:
            await asyncio.gather(*[check_focus_text(focus_text) for focus_text in focus_texts])

        patient_candidates = list(unique_patients)
        if "unknown" in patient_candidates:
            patient_candidates.remove("unknown")

        # Occassionally, multiple patients are referred to in a single string, e.g. "patients 9 and 10" split these out.
        patients_after_splitting: List[str] = []

        async def split_patient(patient):
            if any(term in patient for term in [" and ", " or "]):
                split_response = await self._run_json_prompt(
                    prompt_filepath=self._PROMPTS["split_patients"],
                    params={"patient_list": f'"{patient}"'},  # Encase in double-quotes in prep for bulk calling.
                    prompt_settings={"prompt_tag": "observation__split_patients"},
                )
                patients_after_splitting.extend(split_response.get("patients", []))
            else:
                patients_after_splitting.append(patient)

        await asyncio.gather(*[split_patient(patient) for patient in patient_candidates])

        # If more than 5 (TODO parameterize?) patients are identified, risk of false positives is increased.
        # If there are focus texts (tables), assume lists of patients are available in those tables and cross-check.
        # If there are no focus texts, use the full text of the paper.
        if len(patients_after_splitting) >= 5:
            texts_to_check = focus_texts if focus_texts else [full_text]
            checked_patients = await self._check_patients(patients_after_splitting, texts_to_check)

            if not checked_patients and texts_to_check == focus_texts:
                # All patients failed checking in focus texts, try the full text.
                checked_patients = await self._check_patients(patients_after_splitting, [full_text])
        else:
            checked_patients = patients_after_splitting

        return checked_patients

    async def _find_variant_descriptions(
        self, full_text: str, focus_texts: Sequence[str] | None, gene_symbol: str
    ) -> Sequence[str]:
        """Identify the genetic variants relevant to the gene_symbol described in the full text of the paper.

        Returned variants will be _as described_ in the source text. Downstream manipulations to make them
        HGVS-compliant may be required.
        """
        # Create prompts to find all the unique variants mentioned in the full text and focus texts.
        prompt_runs = [
            self._run_json_prompt(
                prompt_filepath=self._PROMPTS["find_variants"],
                params={"text": text, "gene_symbol": gene_symbol},
                prompt_settings={"prompt_tag": "observation__find_variants"},
            )
            for text in [full_text] + list(focus_texts or [])
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

        # TODO: evaluate tasking the LLM to return an empty list instead of "unknown" when no variants are found.
        candidates = list({_strip_gene_symbol(v) for r in responses for v in r.get("variants", []) if v != "unknown"})

        split_prompt_runs = []
        # If the variant is reported with both coding and protein-level
        # descriptions, split these into two with another prompt.
        for i in reversed(range(len(candidates))):
            if "p." in candidates[i] and "c." in candidates[i]:
                split_prompt_runs.append(
                    self._run_json_prompt(
                        prompt_filepath=self._PROMPTS["split_variants"],
                        params={"variant_list": f'"{candidates[i]}"'},  # Encase in double-quotes for bulk calling.
                        prompt_settings={"prompt_tag": "observation__split_variants"},
                    )
                )
                del candidates[i]
        # Run split prompts in parallel.
        split_responses = await asyncio.gather(*split_prompt_runs)
        # Add the split variants back in to the candidates list.
        candidates.extend(v for r in split_responses for v in r.get("variants", []))
        return candidates

    async def _find_genome_build(self, full_text: str) -> str | None:
        """Identify the genome build used in the paper."""
        response = await self._run_json_prompt(
            prompt_filepath=self._PROMPTS["find_genome_build"],
            params={"text": full_text},
            prompt_settings={"prompt_tag": "observation__find_genome_build"},
        )

        return response.get("genome_build", "unknown")

    async def _link_entities(
        self, full_text: str, patients: Sequence[str], variants: Sequence[str], gene_symbol: str
    ) -> Dict[str, List[str]]:
        params = {
            "text": full_text,
            "patients": ", ".join(patients),
            "variants": ", ".join(variants),
            "gene_symbol": gene_symbol,
        }
        response = await self._run_json_prompt(
            prompt_filepath=self._PROMPTS["link_entities"],
            params=params,
            prompt_settings={"prompt_tag": "observation__link_entities"},
        )

        # TODO, consider validating the links.
        # TODO, consider evaluating focus texts in addition to the full-text.

        return response

    def _create_variant_from_text(
        self, variant_str: str, gene_symbol: str, genome_build: str | None
    ) -> HGVSVariant | None:
        """Attempt to create an HGVSVariant object from a variant description extracted from free text.

        variant_str is a string representation of a variant, but since it's being extracted from paper text it can take
        a variety of formats. It is the responsibility of this method to handle much of this preprocessing and provide
        standardized representations to `self._variant_factory` for parsing.
        """
        # If the variant_str contains a dbsnp rsid, parse it and return the variant.
        if matched := re.match(r"(rs\d+)", variant_str):
            try:
                return self._variant_factory.parse_rsid(matched.group(1))
            except Exception as e:
                logger.warning(f"Unable to create variant from {variant_str} and {gene_symbol}: {e}")
                return None

        # Otherwise, assume we're working with an hgvs-like variant.

        # Remove all the spaces from the variant string.
        variant_str = variant_str.replace(" ", "")

        # Occassionally gene_symbol is embedded in variant_str, if it is, we'll have to extract it.
        # This is generally either of the form gene_symbol:variant or gene_symbol(variant). Sometimes,
        # gene_symbol is prefixed with a 'g' (e.g., pmid:33117677).
        variant_str = re.sub(f"g?{gene_symbol}:", "", variant_str)
        search_result = re.search(r"g?" + gene_symbol + r"\((.*?)\)", variant_str)
        if search_result:
            variant_str = search_result.group(1)

        # Split out the refseq if it's present.
        if variant_str.find(":") >= 0:
            refseq, variant_str = variant_str.split(":", 1)
            refseq = refseq.strip()
            variant_str = variant_str.strip()
        else:
            refseq = None
            variant_str = variant_str.strip()

        # If the refseq looks like a chromosome designation, we've got to figure out the corresponding refseq, which
        # will depend on the genome build.
        if refseq and refseq.find("chr") >= 0:
            refseq = f"{genome_build}({refseq})"

        # Occassionally, protein level descriptions do not include the p. prefix, add it if it's missing.
        # This will only currently handle fairly simple protein level descriptions.
        if re.search(r"^[A-Za-z]+\d+[A-Za-z]", variant_str):
            variant_str = "p." + variant_str

        # Single-letter protein level descriptions should use * for a stop codon, not X.
        variant_str = re.sub(r"(p\.[A-Z]\d+)X", r"\1*", variant_str)

        # Fix c. descriptions that are erroneously written as c.{ref}{pos}{alt} instead of c.{pos}{ref}>{alt}.
        variant_str = re.sub(r"c\.([ACTG])(\d+)([A-Z]+)", r"c.\2\1>\3", variant_str)

        # Fix three-letter p. descriptions that don't follow the capitalization convention.
        # For now, only handle reference AAs and single missense alternate AAs.
        if match := re.match(r"p\.([A-za-z][a-z]{2})(\d+)([A-za-z][a-z]{2})*(.*?)$", variant_str):
            ref_aa, pos, alt_aa, extra = match.groups()
            variant_str = f"p.{ref_aa.capitalize()}{pos}{alt_aa.capitalize() if alt_aa else ''}{extra}"

        # Frameshift should be designated with fs, not frameshift
        variant_str = variant_str.replace("frameshift", "fs")

        try:
            return self._variant_factory.parse(variant_str, gene_symbol, refseq)
        except Exception as e:
            logger.warning(f"Unable to create variant from {variant_str} and {gene_symbol}: {e}")
            return None

    def find_observations(self, gene_symbol: str, paper: Paper) -> Mapping[Tuple[HGVSVariant, str], Mapping[str, Any]]:
        """Identify all observations relevant to `gene_symbol` in `paper`.

        `gene_symbol` should be a gene_symbol. `paper` is the paper to search for relevant observations. Paper must be
        in the PMC-OA dataset and have license terms that permit derivative works based on current restrictions.

        Observations are logically "clinical" observations of a variant in a human, thus this function returns a dict
        keyed by tuples of variants and string representations of the individual in which that variant was observed. The
        values in this dictionary are dictionaries relevant to this observation throughout the paper. Keys in this
        dictionary include, "variant_descriptions", "patient_descriptions", "texts".

        """
        texts = self._get_paper_texts(paper)
        full_text = texts.get("full_text", None)
        table_texts = texts.get("tables", {})

        if not full_text:
            logger.warning(f"Skipping {paper.id} because full text could not be retrieved")
            return {}

        # Determine the candidate genetic variants matching `gene_symbol`
        variant_descriptions = await self._find_variant_descriptions(
            full_text=full_text, focus_texts=list(table_texts.values()), gene_symbol=gene_symbol
        )
        logger.info(f"Found the following variants described for {gene_symbol} in {paper}: {variant_descriptions}")

        # If necessary, determine the genome build most likely used for those variants.
        # TODO: consider doing this on a per-variant bases.
        if any("chr" in v or "g." in v for v in variant_descriptions):
            genome_build = await self._find_genome_build(full_text=full_text)
            logger.info(f"Found the following genome build in {paper}: {genome_build}")
        else:
            genome_build = None

        # Variant objects, keyed by variant description, those that fail to parse are discarded.
        variants = {
            desc: self._create_variant_from_text(desc, gene_symbol, genome_build) for desc in variant_descriptions
        }
        # Note we're keeping invalid variants here.
        variants = {desc: variant for desc, variant in variants.items() if variant is not None}

        if not variants:
            observations = {}
        else:
            # Determine all of the patients specifically referred to in the paper, if any.
            patients = self._find_patients(full_text=full_text, focus_texts=list(table_texts.values()))
            logger.info(f"Found the following patients in {paper}: {patients}")

            # TODO, consider consolidating variants here, before linking with patients.

            # If there are both variants and patients, build a mapping between the two,
            # if there are only variants and no patients, no need to link, just assign all the variants to "unknown".
            # if there are no variants (regardless of patients), then there are no observations to report.
            if patients:
                observations = await self._link_entities(full_text, patients, list(variants.keys()), gene_symbol)
            else:
                observations = {"unknown": list(variants.keys())}

        # TODO, if we've split variant descriptions above, then we run the risk of the observations returning the
        # unsplit variant entity, which will not match the keys in variant objects. Either try to convince the LLM to
        # only use the specific variants we provide, or find a way to be robust to the split during variant object
        # lookup below.

        result: Dict[Tuple[HGVSVariant, str], Dict[str, Any]] = {}
        for individual, variant_strs in observations.items():
            # Consolidate variants within each observation so we only get one variant object per observation.
            variants_to_consolidate: Dict[HGVSVariant, str] = {}
            for variant_str in variant_strs:
                variant = variants.get(variant_str)
                if variant is None:
                    logger.warning(f"Variant {variant_str} not found in variant_objects")
                    continue
                variants_to_consolidate[variant] = variant_str

            if not variants_to_consolidate:
                continue

            consolidation_map = self._variant_comparator.consolidate(
                list(variants_to_consolidate.keys()), disregard_refseq=True
            )

            # Reverse the consolidation map so every key is one of the consolidated variants, and the corresponding
            # value is the variant to which it has been consolidated.
            reverse_consolidation_map = {
                source: target for target, sources in consolidation_map.items() for source in sources
            }

            # Now for all the variants to consolidate, collect their corresponding variant descriptions into a list of
            # strings, keyed by the consolidated variant.
            consolidated_variants: Dict[HGVSVariant, List[str]] = {}
            for variant, variant_desc in variants_to_consolidate.items():
                consolidated_variant = reverse_consolidation_map.get(variant, variant)
                if consolidated_variant not in consolidated_variants:
                    consolidated_variants[consolidated_variant] = []
                consolidated_variants[consolidated_variant].append(variant_desc)
                consolidated_variants[consolidated_variant].append(variant.hgvs_desc)  # TODO, reconsider?

            for variant, variant_descs in consolidated_variants.items():
                payload = {
                    "variant_descriptions": variant_descs,
                    "patient_descriptions": [individual],
                    "texts": [
                        full_text
                    ],  # TODO, consider adding focus_texts here, or find variant/patient specific texts.
                }
                if (variant, individual) in result:
                    logger.warning(f"Duplicate observation for {variant} and {individual} in {paper.id}. Skipping.")
                    continue
                # Only keep variants associated with the "unmatched_variants" individual if they're not already
                # associated with a "real" individual.
                if individual == "unmatched_variants":
                    if any(x[0] == variant for x in result):
                        continue
                    else:
                        result[(variant, "unknown")] = payload
                else:
                    result[(variant, individual)] = payload

        return result
