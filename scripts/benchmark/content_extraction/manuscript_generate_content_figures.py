"""This notebook is intended to be used to generate paper finding benchmark figures for the manuscript."""

# %% Imports.

import os
from typing import Any, Dict, List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from scripts.benchmark.utils import CONTENT_COLUMNS, get_benchmark_run_ids, get_eval_df, load_run

# %% Constants.

OUTPUT_DIR = ".out/manuscript_content_extraction"

# ensure OUTPUT_DIR exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

LOCAL_CONTENT_COLUMNS = CONTENT_COLUMNS.copy()
LOCAL_CONTENT_COLUMNS.remove("study_type")
LOCAL_CONTENT_COLUMNS.remove("animal_model")
LOCAL_CONTENT_COLUMNS.remove("engineered_cells")
LOCAL_CONTENT_COLUMNS.remove("patient_cells_tissues")

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

        for column in LOCAL_CONTENT_COLUMNS:

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


# %% Make another version of the plot where train and test are shown together.

sns.set_theme(style="whitegrid")

all_obs_run_stats_labeled = []
for run_type in ["train", "test"]:
    run_stats_labeled = all_obs_run_stats[run_type].copy()
    run_stats_labeled["split"] = run_type
    all_obs_run_stats_labeled.append(run_stats_labeled)

obs_run_stats_labeled = pd.concat(all_obs_run_stats_labeled)

obs_run_stats_labeled_melted = obs_run_stats_labeled[
    [
        "split",
        "run_id",
        "zygosity",
        "variant_type",
        "variant_inheritance",
        "phenotype",
    ]
].melt(id_vars=["split", "run_id"], var_name="metric", value_name="result")

# Recode split from "train" and "test" to "dev" and "eval".
obs_run_stats_labeled_melted["split"] = obs_run_stats_labeled_melted["split"].map({"train": "dev", "test": "eval"})

plt.figure(figsize=(4, 3))

# Plot the eval data only.
g = sns.barplot(
    data=obs_run_stats_labeled_melted.query("split == 'eval'"),
    x="metric",
    y="result",
    errorbar="sd",
    hue="split",
    alpha=0.6,
    palette={"dev": "#1F77B4", "eval": "#FA621E"},
)
g.xaxis.set_label_text("")
g.yaxis.set_label_text("Accuracy")
g.set_xticklabels(g.get_xticklabels())

# Remove the legend.
g.get_legend().remove()

g.title.set_text("Content extraction")
plt.ylim(0.5, 1)


# Replace all the underscores in the xticklabels with spaces.
def label_fix(label: str) -> str:
    # replace underscores with spaces, and replace "variant" with "v.", capitalize the first letter of each word
    return "\n".join([word.capitalize() for word in label.split("_")])


g.set_xticklabels([label_fix(label.get_text()) for label in g.get_xticklabels()])

plt.savefig(f"{OUTPUT_DIR}/content_extraction_accuracy.png", bbox_inches="tight")

# %% Print them instead.

max_columns = pd.get_option("display.max_columns")
pd.set_option("display.max_columns", None)

width = pd.get_option("display.width")
pd.set_option("display.width", 2000)

for run_type in ["train", "test"]:
    run_stats = all_obs_run_stats[run_type]

    run_type_alt = "dev" if run_type == "train" else "eval"

    print(f"-- Content extraction benchmark results ({run_type_alt}; N={run_stats.shape[0]}) --")

    print(run_stats[["run_id"] + LOCAL_CONTENT_COLUMNS])

    print()
    print(run_stats[LOCAL_CONTENT_COLUMNS].aggregate(["mean", "std"]))
    print()

pd.set_option("display.max_columns", max_columns)
pd.set_option("display.width", width)


# %% Intentionally empty.
