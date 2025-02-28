"""This script explores phenotype comparison results."""

# %% Imports.

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.benchmark.utils import generalize_hpo_term, get_benchmark_run_ids, hpo_str_to_set, load_run

# %% Constants.

TRAIN_RUNS = get_benchmark_run_ids("GPT-4-Turbo", "train")
TEST_RUNS = get_benchmark_run_ids("GPT-4-Turbo", "test")

# %% Functions.


def preprocess_content_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[["phenotype_truth", "phenotype_output", "phenotype_result"]]

    df["phenotype_result"] = df["phenotype_result"].apply(eval)

    df["phenotype_truth"] = df["phenotype_truth"].apply(hpo_str_to_set)
    df["phenotype_output"] = df["phenotype_output"].apply(hpo_str_to_set)

    #df["phenotype_truth_gen"] = df["phenotype_truth"].apply(lambda x: {generalize_hpo_term(term) for term in x})
    #df["phenotype_output_gen"] = df["phenotype_output"].apply(lambda x: {generalize_hpo_term(term) for term in x})

    df["n_truth"] = df["phenotype_truth"].apply(len)
    df["n_output"] = df["phenotype_output"].apply(len)
    #df["n_truth_gen"] = df["phenotype_truth_gen"].apply(len)
    #df["n_output_gen"] = df["phenotype_output_gen"].apply(len)

    return df


# %% Load the data.

all_runs: Dict[str, pd.DataFrame] = {}
all_run_stats: Dict[str, pd.DataFrame] = {}

for run_type, run_ids in [("train", TRAIN_RUNS), ("test", TEST_RUNS)]:

    runs = [load_run(id, "content_extraction", "content_extraction_results.tsv") for id in run_ids]

    run_stats_dicts: List[Dict[str, Any]] = []

    for run_id, raw_run in zip(run_ids, runs):
        if raw_run is None:
            continue

        run = preprocess_content_results(raw_run)
        all_runs[run_id] = run

        run_dict: Dict[str, Any] = {
            "run_id": run_id,
        }

        run_dict["pheno_acc"] = run.phenotype_result.apply(
            lambda x: len(x[2]) == 0 and len(x[3]) == 0
        ).mean()  # perfect
        run_dict["pheno_recall"] = run.phenotype_result.apply(lambda x: len(x[2]) == 0).mean()  # nothing missing
        run_dict["pheno_precision"] = run.phenotype_result.apply(lambda x: len(x[3]) == 0).mean()  # nothing extra

        run_stats_dicts.append(run_dict)

    run_stats = pd.DataFrame(run_stats_dicts)

    all_run_stats[run_type] = run_stats


# %% Make the primary bar plot.

# Make a barplot that shows phenotype accuracy, recall, and precision; separately for train and test runs,
# with error bars showing the standard deviation.

sns.set_theme(style="whitegrid")

all_run_stats_labeled = []
for run_type in ["train", "test"]:
    run_stats_labeled = all_run_stats[run_type].copy()
    run_stats_labeled["split"] = run_type
    all_run_stats_labeled.append(run_stats_labeled)

run_stats_labeled = pd.concat(all_run_stats_labeled)

run_stats_labeled_melted = run_stats_labeled[
    [
        "split",
        "run_id",
        "pheno_acc",
        "pheno_recall",
        "pheno_precision",
    ]
].melt(id_vars=["split", "run_id"], var_name="metric", value_name="value")

plt.figure()
sns.barplot(data=run_stats_labeled_melted.query("split == 'test'"), x="metric", y="value", hue="split", ci="sd")
plt.ylabel("Performance metric")
plt.title("Phenotype extraction performance")
# Replace the xticks with "accuracy", "recall", "precision"
plt.xticks([0, 1, 2], ["Accuracy", "Recall", "Precision"])
plt.xlabel("")

# set the ylim to 0.5-1
plt.ylim(0.5, 1)
plt.legend([], [], frameon=False)

plt.show()

# %% Print the mean and standard deviation of the performance metrics.

for run_type in ["train", "test"]:
    run_stats = all_run_stats[run_type]
    print(f"Run type: {run_type}")
    print(run_stats[["pheno_acc", "pheno_recall", "pheno_precision"]].mean())
    print(run_stats[["pheno_acc", "pheno_recall", "pheno_precision"]].std())
    print()

# %% Separately, make an example scatter plots that shows the effect of interest.

# First, collect the relevant data.
run = all_runs[list(all_runs.keys())[0]]

# Make a scatter plot with n_truth on the x-axis and n_output on the y-axis
# Jitter a little bit to make the points more visible
plt.figure(figsize=(4, 6))
sns.scatterplot(data=run, x="n_truth", y="n_output", alpha=0.5)
plt.xlabel("# of terms in truth")
plt.ylabel("# of terms in output")
plt.title("Phenotype counts in truth vs. output")
# Add black dashed line at x=y
plt.plot([0, 20], [0, 20], "k--")
plt.show()

# Make a scatter plot with n_truth_gen on the x-axis and n_output_gen on the y-axis
# Jitter a little bit to make the points more visible
plt.figure()
sns.scatterplot(data=run, x="n_truth_gen", y="n_output_gen", alpha=0.5)
plt.xlabel("Number of Generalized Phenotypes in Truth")
plt.ylabel("Number of Generalized Phenotypes in Output")
plt.title("Generalized Phenotype Counts in Truth vs. Output")
# Add black dashed line at x=y
plt.plot([0, 10], [0, 10], "k--")
plt.show()

# %% Calculate the mean and std of the number of phenotypes in the truth and output.

for run_type in ["train", "test"]:
    run = all_runs[list(all_runs.keys())[0]]
    print(f"Run type: {run_type}")
    print(run[["n_truth", "n_output"]].mean())
    print(run[["n_truth", "n_output"]].std())
    print(run[["n_truth_gen", "n_output_gen"]].mean())
    print(run[["n_truth_gen", "n_output_gen"]].std())
    print()

# %% Intentionally left blank.
