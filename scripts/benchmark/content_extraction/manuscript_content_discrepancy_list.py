"""This notebook collects a list of all observation discrepancies between run outputs and the truth data."""

# %% Imports.

import os
from typing import Any, Dict, Tuple

import pandas as pd

from scripts.benchmark.utils import CONTENT_COLUMNS, INDICES_FOR_COLUMN, get_benchmark_run_ids, get_eval_df

# %% Constants.

TRAIN_RUNS = get_benchmark_run_ids("GPT-4-Turbo", "train")
TEST_RUNS = get_benchmark_run_ids("GPT-4-Turbo", "test")

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


# %% Generate run stats for observation finding.


def get_row_key(row: pd.Series, col: str) -> Tuple:
    return tuple(row[INDICES_FOR_COLUMN[col]])


dicts: Dict[str, Dict[Tuple[Any, ...], Dict[str, Any]]] = {col: {} for col in CONTENT_COLUMNS}

for run_type, run_ids in [("train", TRAIN_RUNS), ("test", TEST_RUNS)]:

    runs = [load_run(id, "content_extraction_results.tsv") for id in run_ids]

    for run in runs:
        if run is None:
            continue

        run.set_index(["gene", "pmid", "hgvs_desc", "individual_id"], inplace=True)

        for col in CONTENT_COLUMNS:
            eval_df = get_eval_df(run, col).reset_index()

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
dfs = {col: pd.DataFrame(dicts[col]).T for col in CONTENT_COLUMNS}

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
