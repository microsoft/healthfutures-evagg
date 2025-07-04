"""This notebook is intended to be used to generate paper finding benchmark figures for the manuscript.

In addition to this, this notebook also writes out summary TSVs for the benchmark task to be used in subsequent
analyses.

When running this script, you can optionally set the constant `MODEL` below to run the analysis for the canonical
pipeline output set for a different model.
"""

# %% Imports.

import os
from typing import Dict, Set

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from scripts.benchmark.utils import get_benchmark_run_ids, load_run

# %% Constants.


OUTPUT_DIR = ".out/manuscript_paper_finding"
TRUTHSET_VERSIONS = ["v1", "v1.1"]

# ensure OUTPUT_DIR exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


MODEL = "GPT-4-Turbo"  # or "GPT-4o" or "GPT-4o-mini"

TRAIN_RUNS = get_benchmark_run_ids(MODEL, "train")
TEST_RUNS = get_benchmark_run_ids(MODEL, "test")

model_name = f" - {MODEL}"

# %% Compute inter-run-agreement.

for run_type, run_ids in [("train", TRAIN_RUNS), ("test", TEST_RUNS)]:

    runs = [load_run(id, "paper_finding", f"pipeline_mgt_comparison_{TRUTHSET_VERSIONS[0]}.tsv") for id in run_ids]

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


# %% Generate run stats for all truthset versions.
all_run_stats_labeled = []
TRAIN_RUNS = get_benchmark_run_ids(MODEL, "train")
TEST_RUNS = get_benchmark_run_ids(MODEL, "test")

for truthset_version in TRUTHSET_VERSIONS:
    for run_type, run_ids in [("train", TRAIN_RUNS), ("test", TEST_RUNS)]:
        runs = [load_run(id, "paper_finding", f"pipeline_mgt_comparison_{truthset_version}.tsv") for id in run_ids]
        run_stats_dicts = []
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
                    "truthset_version": truthset_version,
                    "split": run_type,
                }
            )
        run_stats = pd.DataFrame(run_stats_dicts)
        run_stats["precision"] = run_stats["n_correct"] / (run_stats["n_correct"] + run_stats["n_irrelevant"])
        run_stats["recall"] = run_stats["n_correct"] / (run_stats["n_correct"] + run_stats["n_missed"])
        run_stats["f1"] = (
            2 * run_stats["precision"] * run_stats["recall"] / (run_stats["precision"] + run_stats["recall"])
        )
        run_stats = run_stats[list(run_stats.columns.drop(["missed", "irrelevant"])) + ["missed", "irrelevant"]]
        all_run_stats_labeled.append(run_stats)


# %%
# set figure size to 3, 3
sns.set_theme(style="whitegrid")

plt.figure(figsize=(3, 3))

run_stats_labeled = pd.concat(all_run_stats_labeled, ignore_index=True)
run_stats_labeled_melted = run_stats_labeled[["split", "run_id", "precision", "recall", "truthset_version"]].melt(
    id_vars=["split", "run_id", "truthset_version"], var_name="stat_type", value_name="stat_value"
)
run_stats_labeled_melted["split"] = run_stats_labeled_melted["split"].map({"train": "dev", "test": "eval"})

g = sns.barplot(
    data=run_stats_labeled_melted.query("split == 'eval'"),
    x="stat_type",
    y="stat_value",
    errorbar="sd",
    hue="truthset_version",
    palette={"v1": "#1F77B4", "v1.1": "#FCA178"},
)
g.xaxis.set_label_text("")
g.yaxis.set_label_text("Proportion")
g.set_title("Paper selection")
g.set_xticklabels([x.get_text().capitalize() for x in g.get_xticklabels()])
g.get_legend().remove()
plt.ylim(0.4, 1)
plt.savefig(f"{OUTPUT_DIR}/paper_finding_benchmark_performance{model_name}.png", bbox_inches="tight")


# %% Print them instead.


# Print tables for each truthset version
for truthset_version in TRUTHSET_VERSIONS:
    print(f"\n==== Results for truthset version: {truthset_version} ====")
    for run_type in ["train", "test"]:
        run_stats = run_stats_labeled[
            (run_stats_labeled["truthset_version"] == truthset_version) & (run_stats_labeled["split"] == run_type)
        ]
        print(f"-- Paper finding benchmark results ({run_type}/{truthset_version}; N={run_stats.shape[0]}) --")
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

for truthset_version in TRUTHSET_VERSIONS:
    for run_type in ["train", "test"]:
        run_stats = run_stats_labeled.query("truthset_version == @truthset_version & split == @run_type")
        run_stats.to_csv(
            f"{OUTPUT_DIR}/paper_finding_benchmarks_{truthset_version}_{run_type}_{MODEL}.tsv", sep="\t", index=False
        )

# %% Intentionally empty.
