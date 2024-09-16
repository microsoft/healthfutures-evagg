"""This notebook is intended to collect a list of all discrepancies between run outputs and the truth data."""

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

OUTPUT_DIR = ".out/manuscript_paper_finding"

# %% Function definitions.


def load_run(run_id: str) -> pd.DataFrame | None:
    """Load the data from a single run."""
    run_file = f".out/run_evagg_pipeline_{run_id}_paper_finding_benchmarks/pipeline_mgt_comparison.csv"
    if not os.path.exists(run_file):
        print(
            f"No benchmark analysis exists for run_id {run_id}. Do you need to run 'manuscript_paper_finding.py' first?"
        )
        return None

    run_data = pd.read_csv(run_file)
    return run_data


@cache
def get_lookup_client() -> IPaperLookupClient:
    ncbi_lookup: IPaperLookupClient = DiContainer().create_instance({"di_factory": "lib/config/objects/ncbi.yaml"}, {})
    return ncbi_lookup


def get_paper(pmid: str) -> Paper | None:
    client = get_lookup_client()
    try:
        return client.fetch(pmid)
    except Exception as e:
        print(f"Error getting title for paper {pmid}: {e}")

    return None


# %% Generate sta

# Build a dataframe listing every (gene, pmid) tuple in either the truth data or the pipeline output.
# If a (gene, pmid) tuple appears in multiple runs, keep track of the number of times it is found.

# Each value in this dictionary is another dictionary with the keys "truth_count", "pipeline_count", and "group".
papers_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}

for run_type, run_ids in [("train", TRAIN_RUNS), ("test", TEST_RUNS)]:

    runs = [load_run(id) for id in run_ids]

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

# Each row in the dataframe where truth_count is not equal to pipeline_count is a discrepancy.
papers["discrepancy"] = papers["truth_count"] != papers["pipeline_count"]


# %% Write the dataframe to a CSV file.
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

papers.to_csv(os.path.join(OUTPUT_DIR, "all_papers_list.csv"))

# %% Write out a text file listing all discrepancies in a random order.

discrepancies = papers.query("discrepancy == True")

# Each discrepancy Should be given a line of text in the output file with the following format:
# True or False, the paper "Here's the paper title" (PMID: 123456) discusses one or more human genetic variants in the
# gene "GENE".

with open(os.path.join(OUTPUT_DIR, "discrepancies.txt"), "w") as f:
    count = 0
    for gene, pmid in discrepancies.sample(frac=0.1, random_state=1).index.values:
        count += 1
        paper = get_paper(pmid)

        if paper is None:
            title = "Unknown title"
            link = "Unknown link"
        else:
            title = paper.props.get("title", "Unknown title")
            link = paper.props.get("link", "Unknown link")

        f.write(
            f'{count}. The paper "{title}" ({link}) '
            f"discusses one or more human genetic variants in the gene {gene}.\n\n"
        )
        f.write("A. True\n")
        f.write("B. False\n\n\n")

# %%
