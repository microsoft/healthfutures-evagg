"""This notebook is intended to be used to generate paper finding benchmark figures for the manuscript."""

# %% Imports.

import os
from typing import Any, Dict, List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# %% Constants.

OUTPUT_DIR = ".out/manuscript_content_extraction"

# TRAIN_RUNS = [
#     "20240909_165847",
#     "20240909_210652",
#     "20240910_044027",
#     "20240910_134659",
#     "20240910_191020",
# ]

# TEST_RUNS = [
#     "20240911_165451",
#     "20240911_194240",
#     "20240911_223218",
#     "20240912_145606",
#     "20240912_181121",
# ]
# MODEL = "GPT-4-Turbo"

# GPT-4o runs
TRAIN_RUNS = ["20240920_080739", "20240920_085154", "20240920_093425", "20240920_101905", "20240920_110151"]
TEST_RUNS = ["20240920_055848", "20240920_062457", "20240920_064935", "20240920_071554", "20240920_074218"]
MODEL = "GPT-4o"

# # GPT-4o-mini runs
# TRAIN_RUNS = ["20240920_165153", "20240920_173754", "20240920_181707", "20240920_185736", "20240920_223702"]
# TEST_RUNS = ["20240920_144637", "20240920_151008", "20240920_153649", "20240920_160020", "20240920_162832"]
# MODEL = "GPT-4o-mini"

model_suffix = f" - {MODEL}"

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

all_obs_run_stats: Dict[str, pd.DataFrame] = {}

for run_type, run_ids in [("train", TRAIN_RUNS), ("test", TEST_RUNS)]:

    runs = [load_run(id, "observation_finding_results.tsv") for id in run_ids]

    obs_run_stats_dicts: List[Dict[str, Any]] = []

    for run_id, run in zip(run_ids, runs):
        if run is None:
            continue

        if run_id == "20240920_074218":
            # This is a known busted run, skip it.
            continue

        precision = run.in_truth[run.in_pipeline].mean()
        recall = run.in_pipeline[run.in_truth].mean()
        n = run.shape[0]

        # Count the variant as being present in truth or pipeline output if it's present in any row, regardless of the
        # individual_id for that row.
        for _, grp_df in run.groupby(["paper_id", "hgvs_desc"]):
            if grp_df.in_truth.any():
                run.loc[grp_df.index, "in_truth"] = True
            if grp_df.in_pipeline.any():
                run.loc[grp_df.index, "in_pipeline"] = True

        run.drop_duplicates(subset=["paper_id", "hgvs_desc"], inplace=True, keep="first")

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


# %% Print them instead.

max_columns = pd.get_option("display.max_columns")
pd.set_option("display.max_columns", None)

width = pd.get_option("display.width")
pd.set_option("display.width", 2000)

for run_type in ["train", "test"]:
    run_stats = all_obs_run_stats[run_type]

    print(f"-- Paper finding benchmark results ({run_type}; N={run_stats.shape[0]}) --")

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
