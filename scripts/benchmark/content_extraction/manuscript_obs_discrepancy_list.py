"""This notebook collects a list of all observation discrepancies between run outputs and the truth data."""

# %% Imports.

import os
from functools import cache
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

obs_dict: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
var_dict: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

for run_type, run_ids in [("train", TRAIN_RUNS), ("test", TEST_RUNS)]:

    runs = [load_run(id, "observation_finding_results.tsv") for id in run_ids]

    for run in runs:
        if run is None:
            continue

        # strip the pmid: prefix from paper_id
        run["paper_id"] = run["paper_id"].str.replace("pmid:", "").astype(int)

        for _, row in run.iterrows():
            obs_key = (row["gene"], row["paper_id"], row["hgvs_desc"], row["individual_id"])
            if obs_key not in obs_dict:
                obs_dict[obs_key] = {
                    "truth_count": int(row.in_truth),
                    "pipeline_count": int(row.in_pipeline),
                    "gene_group": run_type,
                }
            else:
                if row.in_truth:
                    obs_dict[obs_key]["truth_count"] += 1
                if row.in_pipeline:
                    obs_dict[obs_key]["pipeline_count"] += 1

        # Repeat the process ignoring the individual_id, this requires us to drop duplicates, but do so carefully.
        # If in_truth or in_pipeline is True for any row in a group, set it to True for all rows in that group.
        for _, grp_df in run.groupby(["gene", "paper_id", "hgvs_desc"]):
            if grp_df.in_truth.any():
                run.loc[grp_df.index, "in_truth"] = True
            if grp_df.in_pipeline.any():
                run.loc[grp_df.index, "in_pipeline"] = True
        run.drop_duplicates(subset=["gene", "paper_id", "hgvs_desc"], inplace=True, keep="first")

        for _, row in run.iterrows():
            var_key = (row["gene"], row["paper_id"], row["hgvs_desc"])
            if var_key not in var_dict:
                var_dict[var_key] = {
                    "truth_count": int(row.in_truth),
                    "pipeline_count": int(row.in_pipeline),
                    "gene_group": run_type,
                }
            else:
                if row.in_truth:
                    var_dict[var_key]["truth_count"] += 1
                if row.in_pipeline:
                    var_dict[var_key]["pipeline_count"] += 1

# Convert the dictionaries to dataframes.
obs = pd.DataFrame(obs_dict).T
var = pd.DataFrame(var_dict).T

# %% Annotate the dataframes

# For a given (gene, paper, hgvs_desc, individual_id) or (gene, paper, hgvs_desc) the value of `in_truth` is consistent
# across all runs, thus we don't need to bother with finding the discrepancy counts on a per run basis. Instead we can
# determine the number of discrepancies by taking the absolute value of the difference between truth_count and
# pipeline_count.

# Each row in the dataframe where truth_count is not equal to pipeline_count is a discrepancy.
obs["discrepancy_count"] = abs(obs["truth_count"] - obs["pipeline_count"])
obs["discrepancy"] = obs["discrepancy_count"] >= MIN_RECURRENCE

var["discrepancy_count"] = abs(var["truth_count"] - var["pipeline_count"])
var["discrepancy"] = var["discrepancy_count"] >= MIN_RECURRENCE

sns.histplot(data=obs, x="discrepancy_count", bins=range(0, 11), discrete=True)
sns.lineplot(x=[MIN_RECURRENCE - 0.5, MIN_RECURRENCE - 0.5], y=[0, 100], color="red", linestyle="--")
plt.figure()
sns.histplot(data=var, x="discrepancy_count", bins=range(0, 11), discrete=True)
sns.lineplot(x=[MIN_RECURRENCE - 0.5, MIN_RECURRENCE - 0.5], y=[0, 100], color="red", linestyle="--")


print(f"Found {obs['discrepancy'].sum()} observation discrepancies.")
print(f"Found {var['discrepancy'].sum()} variant discrepancies.")

# %% Write the dataframe to a CSV file.
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

obs.to_csv(os.path.join(OUTPUT_DIR, "all_observations_list.csv"))
var.to_csv(os.path.join(OUTPUT_DIR, "all_variants_list.csv"))

# %% Write out a text file listing all OBSERVATION discrepancies in a random order.

discrepancies = obs.query("discrepancy == True")

with open(os.path.join(OUTPUT_DIR, "obs_discrepancies.txt"), "w") as f:
    count = 0
    for gene, pmid, hgvs_desc, individual_id in discrepancies.sample(frac=1, random_state=1).index.values:
        count += 1
        paper = get_paper(pmid)

        if paper is None:
            title = "Unknown title"
            link = "Unknown link"
        else:
            title = paper.props.get("title", "Unknown title")
            link = paper.props.get("link", "Unknown link")

        if individual_id == "inferred proband":
            f.write(
                f'{count}. The paper "{title}" ({link}) '
                f"discusses the variant {hgvs_desc} in {gene} in the primary proband or an unknown/unnamed individual\n"
            )
            f.write("A. This is completely correct\n")
            f.write(f"B. This variant is discussed, but it is not associated with the gene {gene}\n")
            f.write(
                (
                    "C. This variant is discussed, but it is not possessed by the primary "
                    "proband or an unknown/unnamed individual\n\n\n"
                )
            )
        else:
            f.write(
                f'{count}. The paper "{title}" ({link}) '
                f'discusses the variant {hgvs_desc} in {gene} possessed by the individual "{individual_id}"\n'
            )
            f.write("A. This is completely correct\n")
            f.write(f"B. This variant is discussed, but it is not associated with the gene {gene}\n")
            f.write(f'C. This variant is discussed, but it is not possessed by the individual "{individual_id}"\n\n\n')

# %% Intentionally empty.
