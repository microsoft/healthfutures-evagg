"""This notebook is intended to collect a list of all discrepancies between run outputs and the truth data."""

# %% Imports.

import os
from typing import Any, Dict, Tuple

import pandas as pd

from scripts.benchmark.utils import get_benchmark_run_ids, load_run

# %% Constants.

TRAIN_RUNS = get_benchmark_run_ids("GPT-4-Turbo", "train")
TEST_RUNS = get_benchmark_run_ids("GPT-4-Turbo", "test")

# The number of times a discrepancy must appear in order to be included in the output.
# By setting MIN_RECURRENCE to 3, with 5 each TRAIN and TEST runs, we're looking for discrepancies that appear in more
# than half the runs.
MIN_RECURRENCE = 3

OUTPUT_DIR = ".out/manuscript_paper_finding"

# %% Build a dataframe listing every (gene, pmid) tuple in either the truth data or the pipeline output.
# If a (gene, pmid) tuple appears in multiple runs, keep track of the number of times it is found.

# Each value in this dictionary is another dictionary with the keys "truth_count", "pipeline_count", and "group".
papers_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}

for run_type, run_ids in [("train", TRAIN_RUNS), ("test", TEST_RUNS)]:

    runs = [load_run(id, "paper_finding", "pipeline_mgt_comparison.tsv") for id in run_ids]

    for run in runs:
        if run is None:
            continue

        for _, row in run.iterrows():
            key = (row["gene"], row["pmid"])
            if key not in papers_dict:
                papers_dict[key] = {
                    "truth_count": int(row.in_truth),
                    "pipeline_count": int(row.in_pipeline),
                    "gene_group": run_type,
                }
            else:
                if row.in_truth:
                    papers_dict[key]["truth_count"] += 1
                if row.in_pipeline:
                    papers_dict[key]["pipeline_count"] += 1

# Convert the dictionary to a dataframe.
papers = pd.DataFrame(papers_dict).T

# %% Annotate the dataframe.

# For a given (gene, paper) the value of `in_truth` is consistent across all runs, thus we don't need to bother with
# finding the discrepancy counts on a per run basis. Instead we can determine the number of discrepancies by taking the
# absolute value of the difference between truth_count and pipeline_count.

# Each row in the dataframe where truth_count is not equal to pipeline_count is a discrepancy.
papers["discrepancy"] = abs(papers["truth_count"] - papers["pipeline_count"]) >= MIN_RECURRENCE
print(f"Found {papers['discrepancy'].sum()} discrepancies.")
print(f"Found {(~papers['discrepancy']).sum()} agreements.")

# %% Write the dataframe to a CSV file.
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

papers.reset_index(names=["gene", "pmid"]).to_csv(
    os.path.join(OUTPUT_DIR, "all_papers_list.tsv"), sep="\t", index=False
)

# %% Intentionally empty.
