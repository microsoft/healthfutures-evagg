"""This notebook is intended to be used to generate paper finding benchmark figures for the manuscript."""

# %% Imports.

import os
from typing import Any, Dict, List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

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


# %% Generate run stats.

all_run_stats: Dict[str, pd.DataFrame] = {}

for run_type, run_ids in [("train", TRAIN_RUNS), ("test", TEST_RUNS)]:

    runs = [load_run(id) for id in run_ids]

    run_stats_dicts: List[Dict[str, Any]] = []

    for run_id, run in zip(run_ids, runs):
        if run is None:
            continue

        n_correct = run.query("in_truth == True and in_pipeline == True").shape[0]
        missed = run.query("in_truth == True and in_pipeline == False")
        n_missed = missed.shape[0]
        missed_queries = set(missed["pmid_with_query"].unique())
        irrelevant = run.query("in_truth == False and in_pipeline == True")
        n_irrelevant = irrelevant.shape[0]
        irrelevant_queries = set(irrelevant["pmid_with_query"].unique())

        run_stats_dicts.append(
            {
                "run_id": run_id,
                "n_correct": n_correct,
                "n_missed": n_missed,
                "n_irrelevant": n_irrelevant,
                "missed": missed_queries,
                "irrelevant": irrelevant_queries,
            }
        )

    # Make a dataframe out of run_counts_dicts.
    run_stats = pd.DataFrame(run_stats_dicts)

    # Add derived stats.
    run_stats["precision"] = run_stats["n_correct"] / (run_stats["n_correct"] + run_stats["n_irrelevant"])
    run_stats["recall"] = run_stats["n_correct"] / (run_stats["n_correct"] + run_stats["n_missed"])
    run_stats["f1"] = 2 * run_stats["precision"] * run_stats["recall"] / (run_stats["precision"] + run_stats["recall"])

    # Reorder columns, putting missed and irrelevant at the end.
    run_stats = run_stats[list(run_stats.columns.drop(["missed", "irrelevant"])) + ["missed", "irrelevant"]]

    all_run_stats[run_type] = run_stats

# %% Make the counts performance barplot.
sns.set_theme(style="whitegrid")

for run_type in ["train", "test"]:
    run_stats = all_run_stats[run_type]

    # Convert this dataframe to have the columns: run_id, count_type, and count_value.
    run_counts_melted = run_stats[["run_id", "n_correct", "n_missed", "n_irrelevant"]].melt(
        id_vars="run_id", var_name="count_type", value_name="count_value"
    )

    plt.figure()

    g = sns.barplot(
        data=run_counts_melted,
        x="count_type",
        y="count_value",
        errorbar="sd",
        alpha=0.6,
    )
    g.xaxis.set_label_text("")
    g.yaxis.set_label_text("Papers in category")
    g.title.set_text(f"Paper finding benchmark results ({run_type}; N={run_stats.shape[0]})")

# %% Make the stats performance barplot.
sns.set_theme(style="whitegrid")

for run_type in ["train", "test"]:
    run_stats = all_run_stats[run_type]

    # Convert this dataframe to have the columns: run_id, count_type, and count_value.
    run_stats_melted = run_stats[["run_id", "precision", "recall", "f1"]].melt(
        id_vars="run_id", var_name="stat_type", value_name="stat_value"
    )

    plt.figure()

    g = sns.barplot(
        data=run_stats_melted,
        x="stat_type",
        y="stat_value",
        errorbar="sd",
        alpha=0.6,
    )

    g.xaxis.set_label_text("")
    g.yaxis.set_label_text("Performance metric")
    g.title.set_text(f"Paper finding benchmark results ({run_type}; N={run_stats.shape[0]}) - dumb way to display")

# %% Print them instead.

for run_type in ["train", "test"]:
    run_stats = all_run_stats[run_type]

    print(f"-- Paper finding benchmark results ({run_type}; N={run_stats.shape[0]}) --")

    print(run_stats[["n_correct", "n_missed", "n_irrelevant", "precision", "recall", "f1"]])

    print()
    print(run_stats[["n_correct", "n_missed", "n_irrelevant", "precision", "recall", "f1"]].aggregate(["mean", "std"]))
    print()


# %%
