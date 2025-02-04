import asyncio
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from lib.evagg.llm import IPromptClient
from lib.evagg.types import HGVSVariant, ICreateVariants, Paper

from .fulltext import get_fulltext, get_sections
from .interfaces import ICompareVariants, IFindObservations, Observation

PatientVariant = Tuple[HGVSVariant, str]

logger = logging.getLogger(__name__)


def _get_prompt_file_path(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "prompts", "observation", f"{name}.txt")


class ObservationFinder(IFindObservations):
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
        variant_factory: ICreateVariants,
        variant_comparator: ICompareVariants,
    ) -> None:
        self._llm_client = llm_client
        self._variant_factory = variant_factory
        self._variant_comparator = variant_comparator

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
        except json.decoder.JSONDecodeError:
            logger.error(f"Failed to parse response from LLM to {prompt_filepath}: {response}")
            return {}

        return result

    async def _check_patients(self, patient_candidates: Sequence[str], texts_to_check: Sequence[str]) -> List[str]:
        checked_patients: List[str] = []

        async def check_patient(patient: str) -> None:
            for text in texts_to_check:
                validation_response = await self._run_json_prompt(
                    prompt_filepath=_get_prompt_file_path("check_patients"),
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

    async def _find_patients(
        self, full_text: str, focus_texts: Sequence[str] | None, metadata: Dict[str, str]
    ) -> Sequence[str]:
        """Identify the individuals (human subjects) described in the full text of the paper."""
        full_text_response = await self._run_json_prompt(
            prompt_filepath=_get_prompt_file_path("find_patients"),
            params={"text": full_text},
            prompt_settings={"prompt_tag": "observation__find_patients", "prompt_metadata": metadata},
        )

        unique_patients = set(full_text_response.get("patients", []))

        # TODO, logically deduplicate patients here, e.g., if a patient is referred to as both "proband" and "IV-1",
        # we should ask the LLM to determine if these are the same individual.

        async def check_focus_text(focus_text: str) -> None:
            focus_response = await self._run_json_prompt(
                prompt_filepath=_get_prompt_file_path("find_patients"),
                params={"text": focus_text},
                prompt_settings={"prompt_tag": "observation__find_patients", "prompt_metadata": metadata},
            )
            unique_patients.update(focus_response.get("patients", []))

        if focus_texts:
            await asyncio.gather(*[check_focus_text(focus_text) for focus_text in focus_texts])

        patient_candidates = list(unique_patients)
        if "unknown" in patient_candidates:
            patient_candidates.remove("unknown")

        # Occassionally, multiple patients are referred to in a single string, e.g. "patients 9 and 10" split these out.
        patients_after_splitting: List[str] = []

        async def split_patient(patient: str) -> None:
            if any(term in patient for term in [" and ", " or "]):
                split_response = await self._run_json_prompt(
                    prompt_filepath=_get_prompt_file_path("split_patients"),
                    params={"patient_list": f'"{patient}"'},  # Encase in double-quotes in prep for bulk calling.
                    prompt_settings={"prompt_tag": "observation__split_patients", "prompt_metadata": metadata},
                )
                patients_after_splitting.extend(split_response.get("patients", []))
            else:
                patients_after_splitting.append(patient)

        await asyncio.gather(*[split_patient(patient) for patient in patient_candidates])

        # For any numeric patient descriptions, check to see whether the description is a substring of another
        # (non-numeric) patient description. If it is, remove it.
        numeric_patients = [p for p in patients_after_splitting if p.isnumeric()]
        non_numeric_patients = [p for p in patients_after_splitting if not p.isnumeric()]
        for numeric_patient in numeric_patients:
            if any(numeric_patient in non_numeric_patient for non_numeric_patient in non_numeric_patients):
                logger.info(f"Removing {numeric_patient} from list of patients as it is a substring of another.")
                patients_after_splitting.remove(numeric_patient)

        # Deduplicate patients that are case-insensitive matches.
        patients_after_splitting = list({patient.lower() for patient in patients_after_splitting})

        # If more than 5 patients are identified, risk of false positives is increased.
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
        self, full_text: str, focus_texts: Sequence[str] | None, gene_symbol: str, metadata: Dict[str, str]
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

    async def _find_genome_build(self, full_text: str, metadata: Dict[str, str]) -> str | None:
        """Identify the genome build used in the paper."""
        response = await self._run_json_prompt(
            prompt_filepath=_get_prompt_file_path("find_genome_build"),
            params={"text": full_text},
            prompt_settings={"prompt_tag": "observation__find_genome_build", "prompt_metadata": metadata},
        )

        return response.get("genome_build", "unknown")

    async def _link_entities(
        self, full_text: str, patients: Sequence[str], variants: Sequence[str], metadata: Dict[str, str]
    ) -> Dict[str, List[str]]:
        params = {
            "text": full_text,
            "patients": ", ".join(patients),
            "variants": ", ".join(variants),
            "gene_symbol": metadata["gene_symbol"],
        }
        response = await self._run_json_prompt(
            prompt_filepath=_get_prompt_file_path("link_entities"),
            params=params,
            prompt_settings={"prompt_tag": "observation__link_entities", "prompt_metadata": metadata},
        )

        return response

    def _get_fulltext_sections(self, paper: Paper) -> Tuple[str, List[str]]:
        # Get paper texts.
        if not paper.props.get("fulltext_xml"):
            logger.warning(f"Skipping {paper.id} because full text could not be retrieved")
            return "", []

        full_text = get_fulltext(paper.props["fulltext_xml"], exclude=["AUTH_CONT", "ACK_FUND", "COMP_INT", "REF"])
        table_sections = list(get_sections(paper.props["fulltext_xml"], include=["TABLE"]))

        table_ids = {t.id for t in table_sections}
        table_texts = []
        for id in table_ids:
            table_texts.append("\n\n".join([sec.text for sec in table_sections if sec.id == id]))

        return full_text, table_texts

    def _get_text_mentioning_variant(self, paper: Paper, variant_descriptions: Sequence[str], allow_empty: bool) -> str:
        sections = get_sections(paper.props["fulltext_xml"])
        filtered_text = "\n\n".join(
            [section.text for section in sections if any(variant in section.text for variant in variant_descriptions)]
        )
        if not filtered_text and not allow_empty:
            sections = get_sections(paper.props["fulltext_xml"])  # Reset the exhausted generator.
            return "\n\n".join([section.text for section in sections])
        return filtered_text

    def _create_variant_from_text(
        self, variant_str: str, gene_symbol: str, genome_build: str | None
    ) -> HGVSVariant | None:
        """Attempt to create an HGVSVariant object from a variant description extracted from free text.

        variant_str is a string representation of a variant, but since it's being extracted from paper text it can take
        a variety of formats. It is the responsibility of this method to handle much of this preprocessing and provide
        standardized representations to `self._variant_factory` for parsing.
        """
        # If the variant_str contains a dbsnp rsid, parse it and return the variant.
        if matched := re.match(r".*?:?(rs\d+).*?", variant_str):
            try:
                variant = self._variant_factory.parse_rsid(matched.group(1))
                if variant and variant.gene_symbol == gene_symbol:
                    return variant
                else:
                    logger.info(
                        f"dbSNP variant {matched.group(1)} is associated with {variant.gene_symbol}, not {gene_symbol}."
                    )
                    return None
            except ValueError as e:
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

        # If the variant string looks nothing like a variant description, give up.
        if not re.search(r"[A-Za-z]", variant_str):
            logger.warning(f"Variant string '{variant_str}' appears unparsable.")
            return None

        # If the refseq looks like a chromosome designation, we've got to figure out the corresponding refseq, which
        # will depend on the genome build.
        if refseq and refseq.find("chr") >= 0:
            refseq = f"{genome_build}({refseq})"
        # Otherwise, it should begin with NM_, NP_, or NC_, otherwise we'll ignore it.
        elif refseq and not re.match(r"(NM_|NP_|NC_)", refseq):
            logger.info(f"Ignoring potentially invalid refseq: {refseq}")
            refseq = None

        # Remove any parentheses and brackets.
        variant_str = variant_str.replace("(", "").replace(")", "")
        variant_str = variant_str.replace("[", "").replace("]", "")

        # To handle variants where splitting failed, remove everything before the first semicolon.
        variant_str = variant_str.split(";")[0]

        # To handle previxes that weren't removed, remove everything up through the last colon.
        variant_str = variant_str.split(":")[-1]

        # Occassionally, protein level descriptions do not include the p. prefix, add it if it's missing.
        # This will only currently handle fairly simple protein level descriptions.
        if re.search(r"^[A-Za-z]+\d+[A-Za-z]+$", variant_str):
            variant_str = "p." + variant_str

        # Occassionally, coding level descriptions do not include the c. prefix, add it if it's missing.
        # This will only currently handle fairly simple coding level descriptions.
        if re.search(r"^\d+[ACGT]>[ACGT]$", variant_str):
            variant_str = "c." + variant_str
        if re.search(r"^\d+(_\d+)?del[ACGT]*$", variant_str):
            variant_str = "c." + variant_str
        if re.search(r"^\d+ins[ACGT]*$", variant_str):
            variant_str = "c." + variant_str

        # Single-letter protein level descriptions should use * for a stop codon, not X or stop.
        variant_str = re.sub(r"(p\.[A-Z]\d+)X", r"\1*", variant_str)
        variant_str = re.sub(r"(p\.[A-Z]\d+)stop", r"\1*", variant_str)

        # Fix c. descriptions that are erroneously written as c.{ref}{pos}{alt} instead of c.{pos}{ref}>{alt}.
        variant_str = re.sub(r"c\.([ACTG])(\d+)([A-Z]+)", r"c.\2\1>\3", variant_str)

        # Fix three-letter p. descriptions that don't follow the capitalization convention.
        # For now, only handle reference AAs and single missense alternate AAs.
        if "del" not in variant_str:
            if match := re.match(r"p\.([A-Za-z][a-z]{2})(\d+)([A-Za-z][a-z]{2})*(.*?)$", variant_str):
                ref_aa, pos, alt_aa, extra = match.groups()
                variant_str = f"p.{ref_aa.capitalize()}{pos}{alt_aa.capitalize() if alt_aa else ''}{extra}"

        # Frameshift should be designated with fs, not frameshift
        variant_str = variant_str.replace("frameshift", "fs")

        # If there's a hypen that's not surrounded by numbers, remove it.
        variant_str = re.sub(r"(?<!\d)-(?!\d)", "", variant_str)

        # Remove everything after the first occurrence of "fs" if it occurs,
        # HGVS nomenclature gets variable in these cases in practice.
        if "fs" in variant_str:
            variant_str = variant_str.split("fs")[0] + "fs"

        try:
            return self._variant_factory.parse(variant_str, gene_symbol, refseq)
        except ValueError as e:
            # Exception is too broad, determine appropriate exceptions to catch and raise otherwise.
            logger.warning(f"Unable to create variant from {variant_str} and {gene_symbol}: {e}")
            return None

    async def _sanity_check_paper(self, full_text: str, gene_symbol: str, metadata: Dict[str, str]) -> bool:
        try:
            result = await self._run_json_prompt(
                prompt_filepath=_get_prompt_file_path("sanity_check"),
                params={"text": full_text, "gene": gene_symbol},
                prompt_settings={
                    "prompt_tag": "observation__sanity_check",
                    "temperature": 0.5,
                    "prompt_metadata": metadata,
                },
            )
        except Exception as e:  # pragma: no cover
            # This is not ideal, but better handling of content length errors would be a more invasive change that
            # we can save for later. Skipping test coverage for the same reason.
            import openai

            if (
                isinstance(e, openai.BadRequestError)
                and isinstance(e.body, dict)
                and e.body.get("code", "") == "context_length_exceeded"
            ):
                logger.warning(f"Context length exceeded for {metadata['paper_id']}. Skipping.")
                return False
            raise e
        return result.get("relevant", True)  # Default to including the paper.

    async def find_observations(self, gene_symbol: str, paper: Paper) -> Sequence[Observation]:
        """Identify all observations relevant to `gene_symbol` in `paper`.

        `gene_symbol` should be a gene_symbol. `paper` is the paper to search for relevant observations. Paper must be
        in the PMC-OA dataset and have license terms that permit derivative works based on current restrictions.

        The returned observation objects are logically "clinical" observations of a variant in a human. Each object
        describes an individual in which a variant was observed along with the relevant text from the paper.
        """
        # Get the full text of the paper and any focus texts (e.g., tables).
        full_text, table_texts = self._get_fulltext_sections(paper)
        metadata = {"gene_symbol": gene_symbol, "paper_id": paper.id}

        # First, sanity check the paper for mention of genetic variants of interest.
        if not await self._sanity_check_paper(full_text, gene_symbol, metadata):
            logger.info(f"Skipping {paper.id} as it doesn't pass initial check for relevance.")
            return []

        # Determine the candidate genetic variants matching `gene_symbol`
        variant_descriptions = await self._find_variant_descriptions(
            full_text=full_text, focus_texts=table_texts, gene_symbol=gene_symbol, metadata=metadata
        )
        logger.debug(f"Found the following variants described for {gene_symbol} in {paper}: {variant_descriptions}")

        if any("chr" in v or "g." in v for v in variant_descriptions):
            genome_build = await self._find_genome_build(full_text=full_text, metadata=metadata)
            logger.info(f"Found the following genome build in {paper}: {genome_build}")
        else:
            genome_build = None

        # Variant objects, keyed by variant description; those that fail to parse are discarded.
        variants_by_description = {
            description: variant
            for description in variant_descriptions
            if (variant := self._create_variant_from_text(description, gene_symbol, genome_build)) is not None
        }

        # Consolidate the variant objects.
        cons_map = self._variant_comparator.consolidate(list(variants_by_description.values()), disregard_refseq=True)

        # Assess the validity of the relationship between each consolidated variant and the query gene.
        # Do this using the consolidated list of variants to reduce the number of AOAI calls.
        async def _check_variant_gene_relationship(consolidated_variant: HGVSVariant) -> None:
            descriptions = [d for d, v in variants_by_description.items() if v in cons_map[consolidated_variant]]
            mentioning_text = self._get_text_mentioning_variant(paper, descriptions, consolidated_variant.valid)
            warning_text = """
Note that this variant failed validation when considered as part of the gene of interest, so it's likely that the
variant isn't actually associated with the gene. But the possibility of previous error exists, so please check again.
"""
            if mentioning_text:
                response = await self._run_json_prompt(
                    prompt_filepath=_get_prompt_file_path("check_variant"),
                    params={
                        "variant_descriptions": ", ".join(descriptions),
                        "gene_symbol": gene_symbol,
                        "text": mentioning_text,
                        "warning": "" if consolidated_variant.valid else warning_text,
                    },
                    prompt_settings={
                        "prompt_tag": "observation__check_variant_gene_relationship",
                        "prompt_metadata": metadata,
                    },
                )
                if response.get("related", False) is False:
                    for description in descriptions:
                        logger.info(f"Removing {description} from the list of variants.")
                        variants_by_description.pop(description)
            else:
                logger.info(
                    f"No text found mentioning {consolidated_variant} in {paper.id} (checked {descriptions}), "
                    "not removing during variant check."
                )

        await asyncio.gather(*[_check_variant_gene_relationship(v) for v in cons_map])

        # Replace variant objects with their consolidated versions.
        rev_cons_map = {value: key for key, values in cons_map.items() for value in values}
        variants_by_description = {d: rev_cons_map.get(v, v) for d, v in variants_by_description.items()}

        descriptions_by_patient = {}
        # If there are both variants and patients, build a mapping between the two,
        # if there are only variants and no patients, no need to link, just assign all the variants to "unknown".
        # if there are no variants (regardless of patients), then there are no observations to report.
        if variants_by_description:
            # Determine all of the patients specifically referred to in the paper, if any.
            patients = await self._find_patients(full_text=full_text, focus_texts=table_texts, metadata=metadata)
            logger.debug(f"Found the following patients in {paper}: {patients}")
            variant_descriptions = list(variants_by_description.keys())

            if patients:
                descriptions_by_patient = await self._link_entities(full_text, patients, variant_descriptions, metadata)
                # TODO, consider validating returned patients.
            else:
                descriptions_by_patient = {"unknown": variant_descriptions}

        # Assemble the observations.
        observations: List[Observation] = []

        individuals = list(descriptions_by_patient.keys())
        # Ensure "unmatched_variants" is always last in the list.
        if "unmatched_variants" in individuals:
            individuals.remove("unmatched_variants")
            individuals.append("unmatched_variants")

        for individual in individuals:
            variant_descriptions = descriptions_by_patient[individual]
            # LLM should not have returned any patient-linked variants that were not in the input.
            if missing_variants := [d for d in variant_descriptions if d not in variants_by_description]:
                logger.warning(f"Variants '{", ".join(missing_variants)}' not found in paper variants.")
                variant_descriptions = [d for d in variant_descriptions if d not in missing_variants]

            variants: Dict[HGVSVariant, List[str]] = defaultdict(list)
            for description in variant_descriptions:
                variants[variants_by_description[description]].append(description)

            for variant, descriptions in variants.items():
                if any(o.variant == variant and o.individual == individual for o in observations):
                    logger.warning(f"Duplicate observation for {variant} and {individual} in {paper.id}. Skipping.")
                    continue
                if individual == "unmatched_variants":
                    individual = "unknown"
                observations.append(
                    Observation(
                        variant=variant,
                        individual=individual,
                        variant_descriptions=list(set(descriptions)),
                        patient_descriptions=[individual],
                        # Recreate the generator each time.
                        texts=list(
                            get_sections(
                                paper.props["fulltext_xml"], exclude=["AUTH_CONT", "ACK_FUND", "COMP_INT", "REF"]
                            )
                        ),
                        paper_id=paper.id,
                    )
                )

        # Remove redundant observations.
        observations_to_remove = []
        for observation in observations:
            if observation.individual != "unknown":
                continue
            if any(
                observation.variant == other_observation.variant
                for other_observation in observations
                if other_observation.individual != "unknown"
            ):
                logger.info(f"Removing redundant observation {observation.individual}, {observation.variant}.")
                observations_to_remove.append(observation)
        for observation in observations_to_remove:
            observations.remove(observation)

        return observations
