"""This notebook is intended to be used to generate paper finding benchmark figures for the manuscript."""

# %% Imports.

import os
from typing import Any, Dict, List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# %% Constants.

OUTPUT_DIR = ".out/manuscript_paper_finding"

# GPT-4-turbo runs (no model name, since this is the default).
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
MODEL = "GPT-4-Turbo"

# # GPT-4o runs
# TRAIN_RUNS = ["20240920_080739", "20240920_085154", "20240920_093425", "20240920_101905", "20240920_110151"]
# TEST_RUNS = ["20240920_055848", "20240920_062457", "20240920_064935", "20240920_071554", "20240920_074218"]
# MODEL = "GPT-4o"

# # GPT-4o-mini runs
# TRAIN_RUNS = ["20240920_165153", "20240920_173754", "20240920_181707", "20240920_185736", "20240920_223702"]
# TEST_RUNS = ["20240920_144637", "20240920_151008", "20240920_153649", "20240920_160020", "20240920_162832"]
# MODEL = "GPT-4o-mini"

model_name = f" - {MODEL}"

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
    g.title.set_text(f"Paper finding benchmark results ({run_type}; N={run_stats.shape[0]}){model_name}")

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
    g.title.set_text(f"Paper finding benchmark results ({run_type}; N={run_stats.shape[0]}){model_name}")

# %% Make another barplot that shows train and test performance together.

sns.set_theme(style="whitegrid")

all_run_stats_labeled = []
for run_type in ["train", "test"]:
    run_stats_labeled = all_run_stats[run_type].copy()
    run_stats_labeled["split"] = run_type
    all_run_stats_labeled.append(run_stats_labeled)

run_stats_labeled = pd.concat(all_run_stats_labeled)

run_stats_labeled_melted = run_stats_labeled[["split", "run_id", "precision", "recall"]].melt(
    id_vars=["split", "run_id"], var_name="stat_type", value_name="stat_value"
)

plt.figure()

g = sns.barplot(
    data=run_stats_labeled_melted,
    x="stat_type",
    y="stat_value",
    errorbar="sd",
    alpha=0.6,
    hue="split",
)

g.xaxis.set_label_text("")
g.yaxis.set_label_text("Performance metric")
g.title.set_text(f"Paper finding benchmark results{model_name}")

plt.ylim(0, 1)

# %% Print them instead.

for run_type in ["train", "test"]:
    run_stats = all_run_stats[run_type]

    print(f"-- Paper finding benchmark results ({run_type}; N={run_stats.shape[0]}) --")

    print(run_stats[["n_correct", "n_missed", "n_irrelevant", "precision", "recall", "f1"]])

    print()
    print(run_stats[["n_correct", "n_missed", "n_irrelevant", "precision", "recall", "f1"]].aggregate(["mean", "std"]))
    print()


# %% Write outputs for later use.

for run_type in ["train", "test"]:
    run_stats = all_run_stats[run_type]
    run_stats.to_csv(f"{OUTPUT_DIR}/paper_finding_benchmarks_{run_type}_{MODEL}.tsv", sep="\t", index=False)

# %%
