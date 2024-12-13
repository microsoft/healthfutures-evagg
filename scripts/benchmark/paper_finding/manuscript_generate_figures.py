"""This notebook is intended to be used to generate paper finding benchmark figures for the manuscript.

In addition to this, this notebook also writes out summary TSVs for the benchmark task to be used in subsequent
analyses.

When running this script, you can optionally set the constant `MODEL` below to run the analysis for the canonical
pipeline output set for a different model.
"""

# %% Imports.

import os
from typing import Any, Dict, List, Set

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from scripts.benchmark.utils import get_benchmark_run_ids, load_run

# %% Constants.

OUTPUT_DIR = ".out/manuscript_paper_finding"

MODEL = "GPT-4-Turbo"  # or "GPT-4o" or "GPT-4o-mini"

TRAIN_RUNS = get_benchmark_run_ids(MODEL, "train")
TEST_RUNS = get_benchmark_run_ids(MODEL, "test")

model_name = f" - {MODEL}"

# %% Compute inter-run-agreement.

for run_type, run_ids in [("train", TRAIN_RUNS), ("test", TEST_RUNS)]:

    runs = [load_run(id, "paper_finding", "pipeline_mgt_comparison.tsv") for id in run_ids]

    run_type_alt = "train" if run_type == "train" else "eval"

    run_papers: Dict[str, Set[int]] = {}

    for run_id, run in zip(run_ids, runs):
        if run is None:
            continue

        run_papers[run_id] = set(run.pmid.unique())

    # Make a dataframe where every row is a pmid that appears at least once across all runs, and each column is a run
    # with a boolean value for whether that pmid appears in that run.
    all_papers = set.union(*run_papers.values())
    paper_presence = pd.DataFrame({run_id: [pmid in run_papers[run_id] for pmid in all_papers] for run_id in run_ids})

    print(f"Paper consistency ({run_type_alt}): {paper_presence.agg("mean", axis=1).agg("mean")}")


# %% Generate run stats.

all_run_stats: Dict[str, pd.DataFrame] = {}

for run_type, run_ids in [("train", TRAIN_RUNS), ("test", TEST_RUNS)]:

    runs = [load_run(id, "paper_finding", "pipeline_mgt_comparison.tsv") for id in run_ids]

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

    run_type_alt = "train" if run_type == "train" else "eval"

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
    g.title.set_text(f"Paper finding benchmark results ({run_type_alt}; N={run_stats.shape[0]}){model_name}")

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

# Recode split from "train" and "test" to "dev" and "eval".
run_stats_labeled_melted["split"] = run_stats_labeled_melted["split"].map({"train": "dev", "test": "eval"})

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
g.title.set_text("Paper finding")

plt.ylim(0.5, 1)

# %% Print them instead.

for run_type in ["train", "test"]:
    run_stats = all_run_stats[run_type]

    print(f"-- Paper finding benchmark results ({run_type}; N={run_stats.shape[0]}) --")

    print(run_stats[["n_correct", "n_missed", "n_irrelevant", "precision", "recall", "f1"]].round(3))

    print()
    print(
        run_stats[["n_correct", "n_missed", "n_irrelevant", "precision", "recall", "f1"]]
        .aggregate(["mean", "std"])
        .round(3)
    )
    print()


# %% Write outputs for later use.

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for run_type in ["train", "test"]:
    run_stats = all_run_stats[run_type]
    run_stats.to_csv(f"{OUTPUT_DIR}/paper_finding_benchmarks_{run_type}_{MODEL}.tsv", sep="\t", index=False)

# %% Intentionally empty.
