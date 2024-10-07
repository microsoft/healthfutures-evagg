import json
import logging
import os
from functools import cache
from typing import List

import pandas as pd

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
    ncbi_lookup: IPaperLookupClient = DiContainer().create_instance({"di_factory": "lib/config/objects/ncbi.yaml"}, {})
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
