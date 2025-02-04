"""This notebook generates observation benchmark figures for the manuscript.

In addition to this, this notebook also writes out summary TSVs for the benchmark task to be used in subsequent
analyses.

When running this script, you can optionally set the constant `MODEL` below to run the analysis for the canonical
pipeline output set for a different model.
"""

# %% Imports.

import os
from typing import Any, Dict, List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from scripts.benchmark.utils import get_benchmark_run_ids, load_run

# %% Constants.

OUTPUT_DIR = ".out/manuscript_content_extraction"

MODEL = "GPT-4-Turbo"  # or "GPT-4o" or "GPT-4o-mini"

TRAIN_RUNS = get_benchmark_run_ids(MODEL, "train")
TEST_RUNS = get_benchmark_run_ids(MODEL, "test")

model_suffix = f" - {MODEL}"

# %% Generate run stats for observation finding.

all_obs_run_stats: Dict[str, pd.DataFrame] = {}

for run_type, run_ids in [("train", TRAIN_RUNS), ("test", TEST_RUNS)]:

    runs = [load_run(id, "content_extraction", "observation_finding_results.tsv") for id in run_ids]

    obs_run_stats_dicts: List[Dict[str, Any]] = []

    for run_id, run in zip(run_ids, runs):
        if run is None:
            continue

        if run_id == "20240920_074218":
            # This is a known busted run, skip it.
            # More specifically, there was a AOAI generation failure in this run where the limit on output tokens
            # was exceeded. Penalizing the corresponding model for this would be unfair.
            continue

        precision = run.in_truth[run.in_pipeline].mean()
        recall = run.in_pipeline[run.in_truth].mean()
        n = run.shape[0]

        # Count the variant as being present in truth or pipeline output if it's present in any row, regardless of the
        # individual_id for that row.
        for _, grp_df in run.groupby(["pmid", "hgvs_desc"]):
            if grp_df.in_truth.any():
                run.loc[grp_df.index, "in_truth"] = True
            if grp_df.in_pipeline.any():
                run.loc[grp_df.index, "in_pipeline"] = True

        run.drop_duplicates(subset=["pmid", "hgvs_desc"], inplace=True, keep="first")

        precision_variant = run.in_truth[run.in_pipeline].mean()
        recall_variant = run.in_pipeline[run.in_truth].mean()
        n_variant = run.shape[0]

        obs_run_stats_dicts.append(
            {
                "run_id": run_id,
                "n": n,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall),
                "n_variant": n_variant,
                "precision_variant": precision_variant,
                "recall_variant": recall_variant,
                "f1_variant": 2 * precision_variant * recall_variant / (precision_variant + recall_variant),
            }
        )

    run_stats = pd.DataFrame(obs_run_stats_dicts)

    all_obs_run_stats[run_type] = run_stats

# %% Make the performance barplot.
sns.set_theme(style="whitegrid")

for run_type in ["train", "test"]:
    run_stats = all_obs_run_stats[run_type]

    obs_perf_melted = run_stats[["run_id", "precision", "recall"]].melt(
        id_vars="run_id", var_name="metric", value_name="result"
    )

    plt.figure()

    g = sns.barplot(
        data=obs_perf_melted,
        x="metric",
        y="result",
        errorbar="sd",
        alpha=0.6,
    )
    g.xaxis.set_label_text("")
    g.yaxis.set_label_text("Performance")
    g.title.set_text(f"Observation finding benchmark results ({run_type}; N={run_stats.shape[0]}){model_suffix}")

    var_perf_melted = run_stats[["run_id", "precision_variant", "recall_variant"]].melt(
        id_vars="run_id", var_name="metric", value_name="result"
    )
    var_perf_melted["metric"] = var_perf_melted["metric"].map(
        {"precision_variant": "precision", "recall_variant": "recall"}
    )

    plt.figure()

    g = sns.barplot(
        data=var_perf_melted,
        x="metric",
        y="result",
        errorbar="sd",
        alpha=0.6,
    )
    g.xaxis.set_label_text("")
    g.yaxis.set_label_text("Performance")
    g.title.set_text(f"Variant finding benchmark results ({run_type}; N={run_stats.shape[0]}){model_suffix}")


# %% Make another version of the performance barplot where both train and test are shown together.

sns.set_theme(style="whitegrid")

all_obs_run_stats_labeled = []
for run_type in ["train", "test"]:
    run_stats_labeled = all_obs_run_stats[run_type].copy()
    run_stats_labeled["split"] = run_type
    all_obs_run_stats_labeled.append(run_stats_labeled)

obs_run_stats_labeled = pd.concat(all_obs_run_stats_labeled)

obs_run_stats_labeled_melted = obs_run_stats_labeled[
    ["split", "run_id", "precision", "recall", "precision_variant", "recall_variant"]
].melt(id_vars=["split", "run_id"], var_name="metric", value_name="result")

plt.figure()

g = sns.barplot(
    data=obs_run_stats_labeled_melted,
    x="metric",
    y="result",
    errorbar="sd",
    hue="split",
    alpha=0.6,
)
# rename the x labels to precision, recall, precision (variant), recall (variant).
g.set_xticklabels(["Precision", "Recall", "Precision (var)", "Recall (var)"])

g.xaxis.set_label_text("")
g.yaxis.set_label_text("Performance metric")
g.title.set_text(f"Observation finding benchmark results{model_suffix}")
plt.ylim(0.5, 1)

# %% Print them instead.

max_columns = pd.get_option("display.max_columns")
pd.set_option("display.max_columns", None)

width = pd.get_option("display.width")
pd.set_option("display.width", 2000)

for run_type in ["train", "test"]:
    run_stats = all_obs_run_stats[run_type]

    print(f"-- Observation finding benchmark results ({run_type}; N={run_stats.shape[0]}) --")

    print(run_stats[["n", "precision", "recall", "n_variant", "precision_variant", "recall_variant"]])

    print()
    print(
        run_stats[["n", "precision", "recall", "n_variant", "precision_variant", "recall_variant"]].aggregate(
            ["mean", "std"]
        )
    )
    print()

pd.set_option("display.max_columns", max_columns)
pd.set_option("display.width", width)


# %% Write summary outputs for later use.

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for run_type in ["train", "test"]:
    run_stats = all_obs_run_stats[run_type]
    run_stats.to_csv(f"{OUTPUT_DIR}/observation_finding_benchmarks_{run_type}_{MODEL}.tsv", sep="\t", index=False)

# %%
