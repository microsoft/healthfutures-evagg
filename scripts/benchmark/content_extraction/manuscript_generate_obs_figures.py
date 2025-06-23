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

# ensure OUTPUT_DIR exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

MODEL = "GPT-4-Turbo"  # or "GPT-4o" or "GPT-4o-mini"

TRAIN_RUNS = get_benchmark_run_ids(MODEL, "train")
TEST_RUNS = get_benchmark_run_ids(MODEL, "test")

model_suffix = f" - {MODEL}"

TRUTHSET_VERSIONS = ["v1", "v1.1"]

# %% Generate run stats for observation finding.

all_obs_run_stats_labeled: List[pd.DataFrame] = []

for truthset_version in TRUTHSET_VERSIONS:
    for run_type, run_ids in [("train", TRAIN_RUNS), ("test", TEST_RUNS)]:

        runs = [
            load_run(id, "content_extraction", f"observation_finding_results_{truthset_version}.tsv") for id in run_ids
        ]

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

            # Count the variant as being present in truth or pipeline output if it's present in any row, regardless of
            # the individual_id for that row.
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
                    "truthset_version": truthset_version,
                    "split": run_type,
                }
            )

        run_stats = pd.DataFrame(obs_run_stats_dicts)

        all_obs_run_stats_labeled.append(run_stats)


# %% Make another version of the performance barplot where both train and test are shown together.

sns.set_theme(style="whitegrid")

obs_run_stats_labeled = pd.concat(all_obs_run_stats_labeled)

obs_run_stats_labeled_melted = obs_run_stats_labeled[
    ["split", "run_id", "precision", "recall", "precision_variant", "recall_variant", "truthset_version"]
].melt(id_vars=["split", "run_id", "truthset_version"], var_name="metric", value_name="result")

# Recode split from "train" and "test" to "dev" and "eval".
obs_run_stats_labeled_melted["split"] = obs_run_stats_labeled_melted["split"].map({"train": "dev", "test": "eval"})

plt.figure(figsize=(6, 3))

# Plot the eval data only.
g = sns.barplot(
    data=obs_run_stats_labeled_melted.query("split == 'eval'"),
    x="metric",
    y="result",
    errorbar="sd",
    hue="truthset_version",
    palette={"v1": "#1F77B4", "v1.1": "#FCA178"},
)
# Rename the x labels to precision, recall, precision (variant), recall (variant).
g.set_xticklabels(["Precision", "Recall", "Variant\nprecision", "Variant\nrecall"])

g.xaxis.set_label_text("")
g.yaxis.set_label_text("Proportion")
g.title.set_text("Observation finding")
g.get_legend().remove()

plt.ylim(0.4, 1)

plt.savefig(f"{OUTPUT_DIR}/observation_finding_performance{model_suffix}.png", bbox_inches="tight")

# %% Print them instead.

max_columns = pd.get_option("display.max_columns")
pd.set_option("display.max_columns", None)

width = pd.get_option("display.width")
pd.set_option("display.width", 2000)

for truthset_version in TRUTHSET_VERSIONS:
    for run_type in ["train", "test"]:
        run_stats = obs_run_stats_labeled.query(f"truthset_version == '{truthset_version}' and split == '{run_type}'")

        run_type_alt = "dev" if run_type == "train" else "eval"

        print(
            f"-- Observation finding benchmark results ({run_type_alt}/{truthset_version}; N={run_stats.shape[0]}) --"
        )

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

for truthset_version in TRUTHSET_VERSIONS:
    for run_type in ["train", "test"]:
        run_stats = obs_run_stats_labeled.query(f"truthset_version == '{truthset_version}' and split == '{run_type}'")
        run_stats.to_csv(
            f"{OUTPUT_DIR}/observation_finding_benchmarks_{truthset_version}_{run_type}_{MODEL}.tsv",
            sep="\t",
            index=False,
        )

# %%
