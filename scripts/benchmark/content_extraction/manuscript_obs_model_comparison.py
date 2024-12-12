"""This notebook is used to generate figures related to model comparison on observation finding performance.

This notebook can only be run after `manuscript_generate_obs_figures.py has been run for all of the models defined
in the `MODELS` constant below.
"""

# %% Imports.

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

OUTPUT_DIR = ".out/manuscript_content_extraction"

# %% Load in source data.

data_dicts: Dict[str, Dict[str, pd.DataFrame]] = {}

for model in MODELS:
    data_dicts[model] = {}
    for split in ["train", "test"]:
        data_dicts[model][split] = pd.read_csv(
            f"{OUTPUT_DIR}/observation_finding_benchmarks_{split}_{model}.tsv", sep="\t"
        )
        data_dicts[model][split]["model"] = model
        data_dicts[model][split]["split"] = split

# Concatenate the dataframes.
data = pd.concat([data_dicts[model][split] for model in MODELS for split in ["train", "test"]])

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
        plt.title(f"Observation finding {metric} [{split}]")
        plt.ylim(0, 1)

# Make a bar plot that combines train and test into a single plot
for metric in ["recall", "recall_variant"]:
    plt.figure(figsize=(6, 6))
    sns.barplot(
        data=data,
        x="model",
        y=metric,
        errorbar="sd",
        hue="split",
    )
    plt.title(f"Model comparison - Observation finding {metric}")
    plt.ylim(0, 1)

# %% Detailed stats.

for metric in ["recall", "recall_variant"]:
    fit_data = data[[metric, "model", "split"]].copy()
    fit_data.reset_index(names="subject", inplace=True)

    # Impute to handle class balancing issues.
    fit_data.loc[len(fit_data)] = {"subject": 4, metric: None, "model": "GPT-4o", "split": "test"}

    fit_data[metric] = fit_data.groupby(["model", "split"])[metric].transform(lambda x: x.fillna(x.mean()))

    aovrm = AnovaRM(fit_data, metric, "subject", within=["model", "split"])
    res = aovrm.fit()

    # Print the results
    print(res)

    # Since model is significant, we can do a post-hoc comparison using Tukey's HSD test.

    mc = MultiComparison(fit_data[metric], fit_data["model"])
    result = mc.tukeyhsd()
    print(result)

# %% Print out summary statistics.

# Make a table that averages across runs, stratified on model and split, giving average precision and recall for each.
summary = (
    data.groupby(["model", "split"])[["precision", "recall", "precision_variant", "recall_variant"]]
    .agg(["mean", "std"])
    .reset_index()
)

# Just display to 3 sig figs
summary = summary.round(3)

# Don't let pandas split rows over newlines.
pd.set_option("display.expand_frame_repr", False)
print(summary)


# %% Intentionally empty.
