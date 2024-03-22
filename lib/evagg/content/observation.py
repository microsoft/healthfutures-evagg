import json
import logging
import os
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from lib.evagg.llm import IPromptClient
from lib.evagg.ref import INormalizeVariants, IPaperLookupClient
from lib.evagg.types import HGVSVariant, ICreateVariants, Paper

logger = logging.getLogger(__name__)


class ObservationFinder:
    _PROMPTS = {
        "find_patients": os.path.dirname(__file__) + "/prompts/observation/find_patients.txt",
        "find_variants": os.path.dirname(__file__) + "/prompts/observation/find_variants.txt",
        "split_variants": os.path.dirname(__file__) + "/prompts/observation/split_variants.txt",
        "link_entities": os.path.dirname(__file__) + "/prompts/observation/link_entities.txt",
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
        normalizer: INormalizeVariants,
    ) -> None:
        self._llm_client = llm_client
        self._paper_lookup_client = paper_lookup_client
        self._variant_factory = variant_factory
        self._normalizer = normalizer

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

    def _find_patients(self, full_text: str, focus_texts: Sequence[str] | None) -> Sequence[str]:
        """Identify the individuals (human subjects) described in the full text of the paper."""
        full_text_response = self._run_json_prompt(
            prompt_filepath=self._PROMPTS["find_patients"],
            params={"text": full_text},
            prompt_settings={"prompt_tag": "observation__find_patients"},
        )

        unique_patients = set(full_text_response.get("patients", []))

        if focus_texts:
            for focus_text in focus_texts:
                focus_response = self._run_json_prompt(
                    prompt_filepath=self._PROMPTS["find_patients"],
                    params={"text": focus_text},
                    prompt_settings={"prompt_tag": "observation__find_patients"},
                )
                unique_patients.update(focus_response.get("patients", []))

        patients = list(unique_patients)

        # TODO: rationalize and fuzzy dedupe.

        if "unknown" in patients:
            patients.remove("unknown")
        return patients

    def _find_variant_descriptions(
        self, full_text: str, focus_texts: Sequence[str] | None, query: str
    ) -> Sequence[str]:
        """Identify the genetic variants relevant to the query described in the full text of the paper.

        `query` should be a gene symbol.
        Returned variants will be _as described_ in the source text. Downstream manipulations to make them
        HGVS-compliant may be required.
        """
        full_text_response = self._run_json_prompt(
            prompt_filepath=self._PROMPTS["find_variants"],
            params={"text": full_text, "gene_symbol": query},
            prompt_settings={"prompt_tag": "observation__find_variants"},
        )

        unique_variants = set(full_text_response.get("variants", []))

        if focus_texts:
            for focus_text in focus_texts:
                focus_response = self._run_json_prompt(
                    prompt_filepath=self._PROMPTS["find_variants"],
                    params={"text": focus_text, "gene_symbol": query},
                    prompt_settings={"prompt_tag": "observation__find_variants"},
                )
                unique_variants.update(focus_response.get("variants", []))

        # Often, the gene-symbol is provided as a prefix to the variant, remove it.
        # Note: we do additional similar checks later, but it's useful to do it now to remove redundancy.
        def _strip_query(x: str) -> str:
            # If x starts with query, remove that and any subsequent colon.
            if x.startswith(query):
                return x[len(query) :].lstrip(":")
            return x

        candidates = list({_strip_query(x) for x in list(unique_variants)})

        if "unknown" in candidates:
            candidates.remove("unknown")
        if not candidates:
            return candidates

        # Often, the variant is reported with both coding and protein-level descriptions, separate these out to
        # two distinct candidates.
        expanded_candidates: List[str] = []

        for candidate in candidates:
            if candidate.find("p.") >= 0 and candidate.find("c.") >= 0:
                split_response = self._run_json_prompt(
                    prompt_filepath=self._PROMPTS["split_variants"],
                    params={"variant_list": f'"{candidate}"'},  # Encase in double-quotes in prep for bulk calling.
                    prompt_settings={"prompt_tag": "observation__split_variants"},
                )
                expanded_candidates.extend(split_response.get("variants", []))
            else:
                expanded_candidates.append(candidate)

        return expanded_candidates

    def _find_genome_build(self, full_text: str) -> str | None:
        """Identify the genome build used in the paper."""
        response = self._run_json_prompt(
            prompt_filepath=self._PROMPTS["find_genome_build"],
            params={"text": full_text},
            prompt_settings={"prompt_tag": "observation__find_genome_build"},
        )

        return response.get("genome_build", "unknown")

    def _link_entities(
        self, full_text: str, patients: Sequence[str], variants: Sequence[str], query: str
    ) -> Dict[str, List[str]]:
        params = {
            "text": full_text,
            "patients": ", ".join(patients),
            "variants": ", ".join(variants),
            "gene_symbol": query,
        }
        response = self._run_json_prompt(
            prompt_filepath=self._PROMPTS["link_entities"],
            params=params,
            prompt_settings={"prompt_tag": "observation__link_entities"},
        )

        return response

    def _create_variant(self, variant_str: str, gene_symbol: str, genome_build: str | None) -> HGVSVariant | None:
        """Create an HGVSVariant object from `variant_str` and `gene_symbol`."""
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
        if re.search(r"^[A-Za-z]+\d+[A-Za-z]+$", variant_str):
            variant_str = "p." + variant_str

        try:
            return self._variant_factory.parse(variant_str, gene_symbol, refseq)
        except Exception as e:
            logger.warning(f"Unable to create variant from {variant_str} and {gene_symbol}: {e}")
            return None

    def _consolidate_variants(self, variants: Mapping[HGVSVariant, str]) -> Dict[HGVSVariant, List[str]]:
        """Consolidate the Variant objects, retaining all descriptions used.

        The logic below is a little gnarly, so it's worth a short description here. Effectively this method is
        attempting to remove any redundancy in `variants` so any keys in that dict that are biologically "linked" are
        merged, and the string descriptions of those variants that were observed in the paper are retained.

        Note that variants are be biologically linked under the following conditions:
         - if they are the same variant
         - if the protein consequence of one is the the same variant as the other
         - if they have the same hgvs_description and their refseq versions only differ by version number
         - if they have the same hgvs_description and their refseqs are different (TODO: fix this logic)

        Note: this method draws no distinction between valid and invalid variants, invalid variants will not be
        normalized, nor will they have a protein consequence, so they are very likely to be considered distinct from
        other biologically linked variants.
        """
        consolidated_variants: Dict[HGVSVariant, List[str]] = {}

        def _get_primary_refseq(v: HGVSVariant) -> str:
            """Return the primary refseq for the variant."""
            if not v.refseq:
                return ""

            start = v.refseq.find("(")
            end = v.refseq.find(")")
            if start >= 0 and end >= 0:
                return v.refseq[start + 1 : end]
            return v.refseq

        def _get_refseq_version(v: HGVSVariant) -> int:
            """Return the refseq version of the variant, 0 if it's predicted and -1 if it's not present."""
            if not v.refseq:
                return -1
            primary_refseq = _get_primary_refseq(v)
            return (
                int(primary_refseq.split(".")[1]) * int(not v.refseq_predicted) if primary_refseq.find(".") >= 0 else -1
            )

        def _get_refseq_accession(v: HGVSVariant) -> str:
            """Return the refseq accession of the variant, "" if it's not present."""
            return _get_primary_refseq(v).split(".")[0] if v.refseq else ""

        def _score_refseq_completeness(v: HGVSVariant) -> int:
            """Return a score for the completeness refseq for a variant."""
            if not v.refseq:
                return 0
            if v.refseq.find("(") >= 0:
                return 3
            if (v.hgvs_desc.startswith("p.") and v.refseq.startswith("NP_")) or (
                v.hgvs_desc.startswith("c.")
                and (v.refseq.startswith("NM_") or v.refseq.startswith("NG_") or v.refseq.startswith("NC_"))
            ):
                return 2
            return 1

        for variant, description in variants.items():
            found = False
            for saved_variant, saved_descriptions in consolidated_variants.items():
                # - if they are the same variant
                if variant == saved_variant and description not in saved_descriptions:
                    consolidated_variants[saved_variant].append(description)
                    found = True

                # - if the protein consequence of one is the the same variant as the other
                if variant.protein_consequence and variant.protein_consequence == saved_variant:
                    consolidated_variants.pop(saved_variant)
                    saved_descriptions.append(description)
                    consolidated_variants[variant] = saved_descriptions
                    found = True
                if variant == saved_variant.protein_consequence:
                    saved_descriptions.append(description)
                    found = True

                if variant.hgvs_desc == saved_variant.hgvs_desc:
                    # - if they have the same hgvs_description and their refseq versions only differ by version number
                    # In this case keep the variant with the highest observed refseq version. If neither have one, keep
                    # the saved variant. If they both do, also keep the saved variant but log a warning.
                    if _get_refseq_accession(variant) == _get_refseq_accession(saved_variant):
                        saved_version = _get_refseq_version(saved_variant)
                        version = _get_refseq_version(variant)
                        if version > saved_version:
                            consolidated_variants.pop(saved_variant)
                            saved_descriptions.append(description)
                            consolidated_variants[variant] = saved_descriptions
                        else:
                            if saved_version == 0 and version == 0:
                                logger.warning(
                                    f"Both {variant} and {saved_variant} have no refseq "
                                    "version. Keeping {saved_variant}."
                                )
                            saved_descriptions.append(description)
                    # - if they have the same hgvs_description and their refseq versions are different, keep the more
                    # complete one, where more complete to less complete is defined as:
                    #   1) having both a refseq and a selector
                    #   2) having a refseq that is appropriate to the variant description
                    #   3) having a refseq that is not predicted
                    #   4) having a refseq at all
                    # Ties go to the saved_variant.
                    else:
                        if _score_refseq_completeness(variant) > _score_refseq_completeness(saved_variant):
                            consolidated_variants.pop(saved_variant)
                            saved_descriptions.append(description)
                            consolidated_variants[variant] = saved_descriptions
                        else:
                            saved_descriptions.append(description)
                    found = True

                if found:
                    break

            # It's a new variant, save it.
            if not found:
                consolidated_variants[variant] = [description]

        return consolidated_variants

    def find_observations(self, query: str, paper: Paper) -> Mapping[Tuple[HGVSVariant, str], Sequence[str]]:
        """Identify all observations relevant to `query` in `paper`.

        `query` should be a gene_symbol. `paper` is the paper to search for relevant observations. Paper must be in the
        PMC-OA dataset and have license terms that permit derivative works based on current restrictions.

        Observations are logically "clinical" observations of a variant in a human, thus this function returns a dict
        keyed by tuples of variants and string representations of the individual in which that variant was observed. The
        values in this dictionary are a collection of mentions relevant to this observation throughout the paper.
        """
        # Obtain the full-text of the paper.
        # TODO: encapsulate
        full_text = "\n".join(paper.props.get("full_text_sections", []))

        # TODO: encapsulate
        table_texts = {}
        root = paper.props.get("full_text_xml")
        if root is not None:
            for passage in root.findall("./passage"):
                if bool(passage.findall("infon[@key='section_type'][.='TABLE']")):
                    id = passage.find("infon[@key='id']").text
                    if not id:
                        logger.error("No id for table, using None as key")
                    if id not in table_texts:
                        table_texts[id] = passage.find("text").text
                    else:
                        table_texts[id] += "\n" + passage.find("text").text

        if not full_text:
            logger.warning(f"Skipping {paper.id} because full text could not be retrieved")
            return {}

        # Determine all of the patients specifically referred to in the paper, if any.
        patients = self._find_patients(full_text=full_text, focus_texts=list(table_texts.values()))
        logger.info(f"Found the following patients in {paper}: {patients}")

        # Determine the candidate genetic variants matching `query`
        variant_descriptions = self._find_variant_descriptions(
            full_text=full_text, focus_texts=list(table_texts.values()), query=query
        )
        logger.info(f"Found the following variants described for {query} in {paper}: {variant_descriptions}")

        # If necessary, determine the genome build most likely used for those variants.
        # TODO: consider doing this on a per-variant bases.
        if any(v.find("chr") >= 0 or v.find("g.") >= 0 for v in variant_descriptions):
            genome_build = self._find_genome_build(full_text=full_text)
            logger.info(f"Found the following genome build in {paper}: {genome_build}")
        else:
            genome_build = None

        # Variant objects, keyed by variant description, those that fail to parse are discarded.
        variants = {v: self._create_variant(v, query, genome_build) for v in variant_descriptions}
        # Note we're keeping invalid variants here.
        variants = {k: v for k, v in variants.items() if v is not None}

        # If there are both variants and patients, build a mapping between the two,
        # if there are only variants and no patients, no need to link, just assign all the variants to "unknown".
        # if there are no variants (regardless of patients), then there are no observations to report.
        if variant_descriptions and patients:
            observations = self._link_entities(full_text, patients, list(variants.keys()), query)
        elif variant_descriptions:
            observations = {"unknown": list(variants.keys())}
        else:
            observations = {}

        result: Dict[Tuple[HGVSVariant, str], Sequence[str]] = {}
        for individual, variant_strs in observations.items():
            # Consolidate variants within each observation so we only get one variant object per observation,
            # deferring to genomic variants over protein variants. If an observation referrs to a variant_str that
            # isn't in our list of variants, log a warning and drop it.
            variants_to_consolidate: Dict[HGVSVariant, str] = {}
            for variant_str in variant_strs:
                variant = variants.get(variant_str)
                if variant is None:
                    logger.warning(f"Variant {variant_str} not found in variant_objects")
                    continue
                variants_to_consolidate[variant] = variant_str

            if not variants_to_consolidate:
                continue

            consolidated_variants = self._consolidate_variants(variants_to_consolidate)

            for variant, _ in consolidated_variants.items():
                # TODO: use values to find tagged sections, that's why we're leaving the use of '.items()' here.
                if (variant, individual) in result:
                    logger.warning(f"Duplicate observation for {variant} and {individual} in {paper.id}. Skipping.")
                    continue
                # Only keep variants associated with the "unmatched_variants" individual if they're not already
                # associated with a "real" individual.
                if individual == "unmatched_variants":
                    if any(x[0] == variant for x in result):
                        continue
                    else:
                        result[(variant, "unknown")] = [full_text]
                else:
                    result[(variant, individual)] = [full_text]

        return result