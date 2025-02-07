"""This script is used to generate figures related to model comparison on observation finding performance.

This notebook can only be run after `manuscript_generate_figures.py has been run for all of the models defined
in the `MODELS` constant below.
"""

# %% Imports.

import json
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import MultiComparison

# %% Constants.

MODELS = [
    "GPT-4-Turbo",
    "GPT-4o",
    "GPT-4o-mini",
]

OUTPUT_DIR = ".out/manuscript_paper_finding"

# %% Load in source data.

data_dicts: Dict[str, Dict[str, pd.DataFrame]] = {}
for model in MODELS:
    data_dicts[model] = {}
    for split in ["train", "test"]:
        data_dicts[model][split] = pd.read_csv(f"{OUTPUT_DIR}/paper_finding_benchmarks_{split}_{model}.tsv", sep="\t")
        data_dicts[model][split]["model"] = model
        data_dicts[model][split]["split"] = split

# Concatenate the dataframes.
data = pd.concat([data_dicts[model][split] for model in MODELS for split in ["train", "test"]])

# Substitute "dev" for "train" and "eval" for "test" in the split column.
data["split"] = data["split"].replace({"train": "dev", "test": "eval"})

# %% Incorporate runtimes and cost estimates.

INVALID_RUNTIME_IDS = ["20240911_223218", "20240920_223702"]


def get_runtime_sec(run_id: str) -> float:

    metadata = json.load(open(f".out/run_evagg_pipeline_{run_id}/run.json"))
    return metadata["elapsed_secs"]


data["runtime"] = data.run_id.apply(get_runtime_sec)

# Set the runtimes for the invalid runs to NaN.
data.loc[data.run_id.isin(INVALID_RUNTIME_IDS), "runtime"] = None

# %% Make plots.

# Make a bar plot for precision on the test set where there is a different bar for each model, and std dev error bars.
for metric in ["precision", "recall"]:
    for split in ["train", "test"]:
        plt.figure(figsize=(6, 6))
        sns.barplot(
            data=data[data["split"] == split],
            x="model",
            y=metric,
            errorbar="sd",
            hue="model",
        )
        plt.title(f"Paper finding {metric} [{split}]")
        plt.ylim(0, 1)

# Make a bar plot that combines train and test into a single plot
for metric in ["precision", "recall"]:
    plt.figure(figsize=(6, 6))
    sns.barplot(
        data=data,
        x="model",
        y=metric,
        errorbar="sd",
        hue="split",
    )
    plt.title(f"Model comparison - Paper finding {metric}")
    plt.ylim(0.6, 1)

# %% Detailed stats.

# Test for a statistical effect of model using a 2-way anova with repeated measures
# The factors are model and split, the dependent variable is recall.


fit_data = data[["recall", "model", "split"]].copy()
fit_data.reset_index(names="subject", inplace=True)

aovrm = AnovaRM(fit_data, "recall", "subject", within=["model", "split"])
res = aovrm.fit()

# Print the results
print(res)

# Since model is significant, we can do a post-hoc comparison using Tukey's HSD test.

mc = MultiComparison(fit_data["recall"], fit_data["model"])
result = mc.tukeyhsd()
print(result)


# %% Print out summary statistics.

# Make a table that averages across runs, stratified on model and split, giving average precision and recall for each.
summary = data.groupby(["model", "split"])[["precision", "recall", "runtime"]].agg(["mean", "std"]).reset_index()

# Just display to 3 sig figs
summary = summary.round(3)
print(summary)


# %% Intentionally empty.
