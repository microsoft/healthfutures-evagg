"""This notebook collects a list of all observation discrepancies between run outputs and the truth data."""

# %% Imports.

import os
import re
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

                    # Get the HPO ids from phenotype_truth and phenotype_output.
                    # This can be done via regex by collecting all of the instances of HP:\d+.
                    # This is a bit of a hack, but it works.
                    truth_ids = (
                        set(re.findall(r"HP:\d+", row["phenotype_truth"]))
                        if pd.notna(row["phenotype_truth"])
                        else set()
                    )
                    output_ids = (
                        set(re.findall(r"HP:\d+", row["phenotype_output"]))
                        if pd.notna(row["phenotype_output"])
                        else set()
                    )
                    # Get all the elements from truth_ids that aren't also in output_ids.
                    truth_only = truth_ids - output_ids

                    agree_dict = result[1]

                    # For every key in agreement_dict, remove the elements from the corresponding value that are
                    # in truth only.
                    for k in agree_dict:
                        agree_dict[k] = list(set(agree_dict[k]) - truth_only)

                    # There should be no empty lists in agree_dict_raw, let's verify.
                    for k in agree_dict:
                        if len(agree_dict[k]) == 0:
                            print(f"Empty list found in agree_dict_raw for {k}.")

                    # This section is pretty gnarly, so it's worth some comments to explain what's going on.
                    # The idea is to do the necessary bookkeeping that allows us to assess whether the output and
                    # pipeline were consistently in agreement/disagreement across runs. Because the phenotype column
                    # has zero or more elements in it AND because we generalized the specific phenotypes to parent
                    # terms during benchmarking, the logic gets a little convoluted.

                    # dicts[col] (col is always "phenotype" here) is a dict where the key is a representation
                    # of an observation's index. This is a tuple, the form of which is variable depending on the
                    # column, see get_row_key(...). The value is a dict that contains the following keys:
                    # - agree_dict: a dict of the generalized terms that were observed in the pipeline output and
                    #   the specific terms that were used to generate them
                    # - agree_count: a dict of the counts of how many times each of the generalized terms were observed
                    # - truth_dict: a dict of the generalized terms that were observed in the truth data
                    #   and the specific terms that were used to generate them
                    # - truth_count: a dict of the counts of how many times each of the generalized terms were observed
                    # - output_dict: a dict of the generalized terms that were observed in the pipeline output
                    #   and the specific terms that were used to generate them
                    # - output_count: a dict of the counts of how many times each of the generalized terms were observed
                    # - gene_group: the run type (train or test)
                    # - total_count: the total number of times this observation was observed across all runs

                    # If necessary, initialize the ledger for this observation. *_dict are each collections of all of
                    # the general phenotype terms with lists of the specific terms that were used to generate them.
                    # *_count are the counts of how many times each of the generalized terms were observed.
                    if key not in dicts[col]:
                        dicts[col][key] = {
                            "agree_dict": agree_dict,
                            "agree_count": dict.fromkeys(agree_dict.keys(), 1),
                            "truth_dict": result[2],
                            "truth_count": dict.fromkeys(result[2].keys(), 1),
                            "output_dict": result[3],
                            "output_count": dict.fromkeys(result[3].keys(), 1),
                            "gene_group": run_type,
                            "total_count": 1,
                        }
                    # If the observation is already in the ledger, update it.
                    else:
                        dicts[col][key]["total_count"] += 1
                        # Each of these for loops keeps track of the specific terms that contributed to generalized
                        # terms as well as the number of times that generalized term was observed. We keep separate
                        # tallies for agreements (generalized terms in both truth and pipeline output), truth only,
                        # and pipeline output only.
                        for k in agree_dict:
                            if k not in dicts[col][key]["agree_dict"]:
                                dicts[col][key]["agree_dict"][k] = list(set(agree_dict[k]))
                                dicts[col][key]["agree_count"][k] = 1
                            else:
                                dicts[col][key]["agree_dict"][k] += agree_dict[k]
                                dicts[col][key]["agree_dict"][k] = list(set(dicts[col][key]["agree_dict"][k]))
                                dicts[col][key]["agree_count"][k] += 1
                        for k in result[2]:
                            if k not in dicts[col][key]["truth_dict"]:
                                dicts[col][key]["truth_dict"][k] = result[2][k]
                                dicts[col][key]["truth_count"][k] = 1
                            else:
                                dicts[col][key]["truth_dict"][k] += result[2][k]
                                dicts[col][key]["truth_dict"][k] = list(set(dicts[col][key]["truth_dict"][k]))
                                dicts[col][key]["truth_count"][k] += 1
                        for k in result[3]:
                            if k not in dicts[col][key]["output_dict"]:
                                dicts[col][key]["output_dict"][k] = result[3][k]
                                dicts[col][key]["output_count"][k] = 1
                            else:
                                dicts[col][key]["output_dict"][k] += result[3][k]
                                dicts[col][key]["output_dict"][k] = list(set(dicts[col][key]["output_dict"][k]))
                                dicts[col][key]["output_count"][k] += 1
                else:
                    if key not in dicts[col]:
                        dicts[col][key] = {
                            "total_count": 1,
                            "agree_count": int(row[f"{col}_result"]),
                            "gene_group": run_type,
                            "truth_value": row[f"{col}_truth"],
                        }
                    else:
                        assert dicts[col][key]["truth_value"] == row[f"{col}_truth"], "Truth values do not match."
                        dicts[col][key]["total_count"] += 1
                        if row[f"{col}_result"]:
                            dicts[col][key]["agree_count"] += 1

# Convert the dictionaries to dataframes.
dfs = {col: pd.DataFrame(dicts[col]).T for col in CONTENT_COLUMNS}

# %% Annotate the dataframes

for col, df in dfs.items():
    if col in ["animal_model", "engineered_cells", "patient_cells_tissues", "study_type"]:
        continue

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

        # Counting discrepancies for phenotype is a little harder, for each observation that is marked as a discrepancy,
        # count all the unique specific terms that are in truth_dict and output_dict.
        #
        # Counting the agreements is similar, though you can have agreements in rows marked as discrepancies, so in this
        # case, you count every unique specific term in agree_dict.
        discrepant_term_count = 0
        agree_term_count = 0
        for _, row in df.iterrows():
            if row.total_count < 2:
                continue

            agree_terms = {term for value in row.agree_dict.values() for term in value}
            agree_term_count += len(agree_terms)

            if row.discrepancy is False:
                continue

            output_discrepant_terms = {term for value in row.output_dict.values() for term in value}
            truth_discrepant_terms = {term for value in row.truth_dict.values() for term in value}
            discrepant_term_count += len(output_discrepant_terms) + len(truth_discrepant_terms)

        print(f"Found {discrepant_term_count} discrepancies for {col}.")
        print(f"Found {agree_term_count} agreements for {col}.")

    else:
        df["discrepancy"] = ((df["total_count"] - df["agree_count"]) >= (df["total_count"] / 2)) & (
            df["total_count"] >= 2
        )
        print(f"Found {df['discrepancy'].sum()} discrepancies for {col}.")
        print(f"Found {(~df['discrepancy']).sum()} agreements for {col}.")

# %% Write the dataframes to a CSV file.
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for col, df in dfs.items():
    df.reset_index(names=INDICES_FOR_COLUMN[col]).to_csv(
        os.path.join(OUTPUT_DIR, f"all_{col}_list.tsv"), sep="\t", index=False
    )

# %% Intentionally empty.
