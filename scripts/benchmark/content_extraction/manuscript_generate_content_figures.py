"""This notebook is intended to be used to generate paper finding benchmark figures for the manuscript."""

# %% Imports.

from typing import Any, Dict, List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from scripts.benchmark.utils import CONTENT_COLUMNS, get_benchmark_run_ids, get_eval_df, load_run

# %% Constants.

TRAIN_RUNS = get_benchmark_run_ids("GPT-4-Turbo", "train")
TEST_RUNS = get_benchmark_run_ids("GPT-4-Turbo", "test")

# %% Generate run stats for content extraction.

all_obs_run_stats: Dict[str, pd.DataFrame] = {}

for run_type, run_ids in [("train", TRAIN_RUNS), ("test", TEST_RUNS)]:

    runs = [load_run(id, "content_extraction", "content_extraction_results.tsv") for id in run_ids]

    obs_run_stats_dicts: List[Dict[str, Any]] = []

    for run_id, run in zip(run_ids, runs):
        if run is None:
            continue

        run.set_index(["gene", "pmid", "hgvs_desc", "individual_id"], inplace=True)

        run_dict: Dict[str, Any] = {
            "run_id": run_id,
        }

        for column in CONTENT_COLUMNS:

            eval_df = get_eval_df(run, column)

            if column == "phenotype":
                result_tuples = eval_df.phenotype_result.apply(eval)
                result = sum(len(t[2]) == 0 and len(t[3]) == 0 for t in result_tuples) / len(result_tuples)
            else:
                result = eval_df[f"{column}_result"].mean()

            run_dict[f"{column}_n"] = eval_df.shape[0]
            run_dict[column] = result

        obs_run_stats_dicts.append(run_dict)

    run_stats = pd.DataFrame(obs_run_stats_dicts)

    all_obs_run_stats[run_type] = run_stats


# %% Make the performance barplot.
sns.set_theme(style="whitegrid")

for run_type in ["train", "test"]:
    run_stats = all_obs_run_stats[run_type]

    content_perf_melted = run_stats[["run_id"] + CONTENT_COLUMNS].melt(
        id_vars="run_id", var_name="field", value_name="accuracy"
    )

    plt.figure()

    g = sns.barplot(
        data=content_perf_melted,
        x="field",
        y="accuracy",
        errorbar="sd",
        alpha=0.6,
    )
    g.xaxis.set_label_text("")
    g.yaxis.set_label_text("Accuracy")
    g.title.set_text(f"Content extraction benchmark results ({run_type}; N={run_stats.shape[0]})")

    # rotate the x-axis labels
    for item in g.get_xticklabels():
        item.set_rotation(90)


# %% Print them instead.

max_columns = pd.get_option("display.max_columns")
pd.set_option("display.max_columns", None)

width = pd.get_option("display.width")
pd.set_option("display.width", 2000)

for run_type in ["train", "test"]:
    run_stats = all_obs_run_stats[run_type]

    print(f"-- Content extraction benchmark results ({run_type}; N={run_stats.shape[0]}) --")

    print(run_stats[["run_id"] + CONTENT_COLUMNS])

    print()
    print(run_stats[CONTENT_COLUMNS].aggregate(["mean", "std"]))
    print()

pd.set_option("display.max_columns", max_columns)
pd.set_option("display.width", width)


# %% Intentionally empty.