"""This script is used to generate figures related to model comparison on observation finding performance.

This notebook can only be run after `manuscript_generate_figures.py has been run for all of the models defined
in the `MODELS` constant below.
"""

# %% Imports.

from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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


# %% Intentionally empty.
