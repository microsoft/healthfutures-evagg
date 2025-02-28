import json
import logging
import os
import re
from functools import cache
from typing import List, Set, Type

import pandas as pd
from pyhpo import HPOTerm, Ontology

from lib.di import DiContainer
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import Paper

logger = logging.getLogger(__name__)


RUN_LIST = "scripts/benchmark/benchmark_runs.json"


def get_benchmark_run_ids(model: str, group: str) -> List[str]:
    """Get the run IDs for a given model and group from the benchmark runs list."""
    with open(RUN_LIST, "r") as f:
        run_dict = json.load(f)

    run_ids = [run["date"] for run in run_dict["runs"] if run["model"] == model and run["group"] == group]
    if not run_ids:
        logger.warning(f"No runs found for model '{model}' and group '{group}'")

    return run_ids


def load_run(run_id: str, task: str, run_filename: str) -> pd.DataFrame | None:
    """Load the data from a single run."""
    run_file = f".out/run_evagg_pipeline_{run_id}_{task}_benchmarks/{run_filename}"
    if not os.path.exists(run_file):
        logger.error(
            f"No benchmark analysis exists for run_id {run_file}. "
            "Do you need to run the corresponding generation script first?"
        )
        return None

    run_data = pd.read_csv(run_file, sep="\t")
    return run_data


@cache
def _get_lookup_client() -> IPaperLookupClient:
    ncbi_lookup: IPaperLookupClient = DiContainer().create_instance(
        {"di_factory": "lib/config/objects/ncbi_cache.yaml"}, {}
    )
    return ncbi_lookup


@cache
def get_paper(pmid: str) -> Paper | None:
    client = _get_lookup_client()
    try:
        return client.fetch(pmid)
    except Exception as e:
        print(f"Error getting title for paper {pmid}: {e}")

    return None


CONTENT_COLUMNS = [
    "animal_model",
    "engineered_cells",
    "patient_cells_tissues",
    "phenotype",
    "study_type",
    "variant_inheritance",
    "variant_type",
    "zygosity",
]

INDICES_FOR_COLUMN = {
    "animal_model": ["gene", "pmid", "hgvs_desc", "individual_id"],
    "engineered_cells": ["gene", "pmid", "hgvs_desc", "individual_id"],
    "patient_cells_tissues": ["gene", "pmid", "hgvs_desc", "individual_id"],
    "phenotype": ["gene", "pmid", "hgvs_desc", "individual_id"],
    "study_type": ["gene", "pmid"],
    "variant_inheritance": ["gene", "pmid", "hgvs_desc", "individual_id"],
    "variant_type": ["gene", "pmid", "hgvs_desc"],
    "zygosity": ["gene", "pmid", "hgvs_desc", "individual_id"],
}


def get_eval_df(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df.empty:
        return df

    indices = INDICES_FOR_COLUMN[column]
    eval_df = df[~df.reset_index().set_index(indices).index.duplicated(keep="first")]
    return eval_df[[f"{column}_result", f"{column}_truth", f"{column}_output"]]


@cache
def _get_ontology() -> Type[Ontology]:
    Ontology()
    return Ontology


def _get_terms_for_depth_recurse(
    target_depth: int, current_depth: int, cur_obj: HPOTerm, forgettable: Set[HPOTerm]
) -> List[str]:
    """Get all HPO terms at a given depth."""
    if current_depth == target_depth:
        if cur_obj in forgettable:
            return []
        return [cur_obj.__str__()]

    forgettable.add(cur_obj)

    return [
        term
        for child in cur_obj.children
        for term in _get_terms_for_depth_recurse(target_depth, current_depth + 1, child, forgettable)
    ]


@cache
def _get_terms_for_depth(depth: int) -> Set[str]:
    """Get all HPO terms at a given depth."""
    return set(
        _get_terms_for_depth_recurse(
            target_depth=depth, current_depth=1, cur_obj=_get_ontology().get_hpo_object("HP:0000001"), forgettable=set()
        )
    )


def generalize_hpo_term(hpo_term: str, depth: int = 3) -> List[str]:
    """Take an HPO term ID and return the generalized version of that term at `depth`.

    `depth` determines the degree to which hpo_term gets generalized, setting depth=1 will always return HP:0000001.
    Note that multiple possible generalizations may exist, this function return all of them. The `depth` of any given
    generalization will be defined as the shortest possible path from the root to that generalization.

    If the provided term is more generalized than `depth` (e.g., "HP:0000118"), i.e., if it has a shorter path to the
    root node, then that term itself will be returned.

    If the provided term doesn't exist in the ontology, then an error will be raised.
    """
    potential_terms = _get_terms_for_depth(depth)

    try:
        hpo_obj = _get_ontology().get_hpo_object(hpo_term)
    except RuntimeError:
        # HPO term not found in pyhpo, can't use
        print("Warning: HPO term not found in pyhpo, can't use", hpo_term)
        return [""]

    parents = potential_terms & {p.__str__() for p in hpo_obj.all_parents}
    if not parents:
        return [hpo_obj.__str__()]
    return [parent for parent in parents]


def hpo_str_to_set(hpo_compound_string: str) -> Set[str]:
    """Convert a delimited list of HPO terms to a set of terms.

    Takes a string of the form "Foo (HP:1234), Bar (HP:4321) and provides a set of strings that correspond to the
    HPO IDs embedded in the string.
    """
    return set(re.findall(r"HP:\d+", hpo_compound_string)) if pd.isna(hpo_compound_string) is False else set()
