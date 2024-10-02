"""This notebook collects a list of all observation discrepancies between run outputs and the truth data."""

# %% Imports.

import os
from functools import cache
from typing import Any, Dict, Tuple

import pandas as pd

from lib.di import DiContainer
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import Paper

# %% Constants.

TRAIN_RUNS = [
    "20240909_165847",
    "20240909_210652",
    "20240910_044027",
    "20240910_134659",
    "20240910_191020",
]

TEST_RUNS = [
    "20240911_165451",
    "20240911_194240",
    "20240911_223218",
    "20240912_145606",
    "20240912_181121",
]

COLUMNS_OF_INTEREST = [
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
    "animal_model": ["gene", "pmid", "individual_id"],
    "engineered_cells": ["gene", "pmid", "individual_id"],
    "patient_cells_tissues": ["gene", "pmid", "individual_id"],
    "phenotype": ["gene", "pmid", "hgvs_desc", "individual_id"],
    "study_type": ["gene", "pmid"],
    "variant_inheritance": ["gene", "pmid", "hgvs_desc", "individual_id"],
    "variant_type": ["gene", "pmid", "hgvs_desc"],
    "zygosity": ["gene", "pmid", "hgvs_desc", "individual_id"],
}

# The number of times a discrepancy must appear in order to be included in the output.
# By setting MIN_RECURRENCE to 3, with 5 each TRAIN and TEST runs, we're looking for discrepancies that appear in more
# than half the runs.
MIN_RECURRENCE = 3

OUTPUT_DIR = ".out/manuscript_content_extraction"

# %% Function definitions.


def load_run(run_id: str, run_filename: str) -> pd.DataFrame | None:
    """Load the data from a single run."""
    run_file = f".out/run_evagg_pipeline_{run_id}_content_extraction_benchmarks/{run_filename}"
    if not os.path.exists(run_file):
        print(
            f"No benchmark analysis exists for run_id {run_file}. "
            "Do you need to run 'manuscript_content_extraction.py' first?"
        )
        return None

    run_data = pd.read_csv(run_file, sep="\t")
    return run_data


def get_eval_df(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df.empty:
        return df

    indices = INDICES_FOR_COLUMN[column]
    eval_df = df[~df.reset_index().set_index(indices).index.duplicated(keep="first")]
    return eval_df[indices + [f"{column}_result", f"{column}_truth", f"{column}_output"]]


@cache
def get_lookup_client() -> IPaperLookupClient:
    ncbi_lookup: IPaperLookupClient = DiContainer().create_instance({"di_factory": "lib/config/objects/ncbi.yaml"}, {})
    return ncbi_lookup


@cache
def get_paper(pmid: str) -> Paper | None:
    client = get_lookup_client()
    try:
        return client.fetch(pmid)
    except Exception as e:
        print(f"Error getting title for paper {pmid}: {e}")

    return None


# %% Generate run stats for observation finding.


def get_row_key(row: pd.Series, col: str) -> Tuple:
    return tuple(row[INDICES_FOR_COLUMN[col]])


dicts: Dict[str, Dict[Tuple[Any, ...], Dict[str, Any]]] = {col: {} for col in COLUMNS_OF_INTEREST}

for run_type, run_ids in [("train", TRAIN_RUNS), ("test", TEST_RUNS)]:

    runs = [load_run(id, "content_extraction_results.tsv") for id in run_ids]

    for run in runs:
        if run is None:
            continue

        print(f"{run_type} - {run.paper_id.nunique()} papers")

        # strip the pmid: prefix from paper_id and turn the column into pmid
        run["pmid"] = run["paper_id"].str.replace("pmid:", "").astype(int)
        run.drop("paper_id", axis=1, inplace=True)

        for col in COLUMNS_OF_INTEREST:
            eval_df = get_eval_df(run, col)

            for _, row in eval_df.iterrows():
                key = get_row_key(row, col)

                if col == "phenotype":
                    result = eval(str(row["phenotype_result"]))
                    if key not in dicts[col]:
                        dicts[col][key] = {
                            "truth_dict": result[2],
                            "truth_count": dict.fromkeys(result[2].keys(), 1),
                            "output_dict": result[3],
                            "output_count": dict.fromkeys(result[3].keys(), 1),
                            "gene_group": run_type,
                            "total_count": 1,
                        }
                    else:
                        dicts[col][key]["total_count"] += 1
                        for k in result[2]:
                            if k not in dicts[col][key]["truth_dict"]:
                                dicts[col][key]["truth_dict"][k] = result[2][k]
                                dicts[col][key]["truth_count"][k] = 1
                            else:
                                dicts[col][key]["truth_dict"][k] += result[2][k]
                                dicts[col][key]["truth_count"][k] += 1
                        for k in result[3]:
                            if k not in dicts[col][key]["output_dict"]:
                                dicts[col][key]["output_dict"][k] = result[3][k]
                                dicts[col][key]["output_count"][k] = 1
                            else:
                                dicts[col][key]["output_dict"][k] += result[3][k]
                                dicts[col][key]["output_count"][k] += 1
                else:
                    if key not in dicts[col]:
                        dicts[col][key] = {
                            "total_count": 1,
                            "agree_count": int(row[f"{col}_result"]),
                            "gene_group": run_type,
                        }
                    else:
                        dicts[col][key]["total_count"] += 1
                        if row[f"{col}_result"]:
                            dicts[col][key]["agree_count"] += 1

# Convert the dictionaries to dataframes.
dfs = {col: pd.DataFrame(dicts[col]).T for col in COLUMNS_OF_INTEREST}

# %% Annotate the dataframes

for col, df in dfs.items():
    if col == "phenotype":
        # Define an output discrepancy for a row as having any value in output_count be greater than or equal to half
        # of the total_count for that row.
        df["output_discrepancy"] = df.apply(
            lambda row: any(
                (row["output_count"][k] >= row["total_count"] / 2) & (row["total_count"] >= 2)
                for k in row["output_count"]
            ),
            axis=1,
        )
        # Same thing for truth discrepancies.
        df["truth_discrepancy"] = df.apply(
            lambda row: any(
                (row["truth_count"][k] >= row["total_count"] / 2) & (row["total_count"] >= 2)
                for k in row["truth_count"]
            ),
            axis=1,
        )
        # If either the output or truth discrepancy is true, then the phenotype discrepancy is true.
        df["discrepancy"] = df["output_discrepancy"] | df["truth_discrepancy"]
    else:
        df["discrepancy"] = ((df["total_count"] - df["agree_count"]) >= (df["total_count"] / 2)) & (
            df["total_count"] >= 2
        )
    print(f"Found {df['discrepancy'].sum()} discrepancies for {col}.")


# %% Write the dataframes to a CSV file.
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for col, df in dfs.items():
    df.reset_index(names=INDICES_FOR_COLUMN[col]).to_csv(
        os.path.join(OUTPUT_DIR, f"all_{col}_list.tsv"), sep="\t", index=False
    )

# %% Intentionally empty.
