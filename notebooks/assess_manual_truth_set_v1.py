"""Notebook for generating figures that assess the v1 manual truth set."""

# %% Imports.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %% Constants.

TRAIN_TSV = "/home/azureuser/repos/ev-agg-exp/data/v1/evidence_train_v1.tsv"
TEST_TSV = "/home/azureuser/repos/ev-agg-exp/data/v1/evidence_test_v1.tsv"

# %% Load the data and concatenate into a single dataframe.

train_df = pd.read_csv(TRAIN_TSV, sep="\t")
test_df = pd.read_csv(TEST_TSV, sep="\t")

df = pd.concat([train_df, test_df])

# recode engineered_cells, patient_cells_tissues, and animal_model to be yes/no instead of x/None.
df["engineered_cells"] = df["engineered_cells"].apply(lambda x: "yes" if x == "x" else "no")
df["patient_cells_tissues"] = df["patient_cells_tissues"].apply(lambda x: "yes" if x == "x" else "no")
df["animal_model"] = df["animal_model"].apply(lambda x: "yes" if x == "x" else "no")

# recode in_supplement to be yes/no instead of Y/N
df["in_supplement"] = df["in_supplement"].apply(lambda x: "yes" if x == "Y" else "no")

# %% Gene-level plots

gene_categorical_columns = ["evidence_base"]

for col in gene_categorical_columns:

    normalized_value_counts = df.groupby(["gene"]).first().groupby("group")[col].value_counts(normalize=True)
    normalized_value_counts = normalized_value_counts.rename("proportion").reset_index()

    mplot = sns.barplot(data=normalized_value_counts, x=col, y="proportion", hue="group")
    mplot.set_xticklabels(mplot.get_xticklabels(), rotation=90)
    plt.title(col)
    plt.show()

# %% Paper-level plots

paper_categorical_columns = ["is_pmc_oa", "license", "study_type"]

for col in paper_categorical_columns:
    # Only counting once for each paper, get a dataframe of value_counts of is_pmc_oa on a per-paper basis, stratified on group.

    normalized_value_counts = df.groupby(["paper_id"]).first().groupby("group")[col].value_counts(normalize=True)
    normalized_value_counts = normalized_value_counts.rename("proportion").reset_index()

    mplot = sns.barplot(data=normalized_value_counts, x=col, y="proportion", hue="group")
    mplot.set_xticklabels(mplot.get_xticklabels(), rotation=90)
    plt.title(col)
    plt.show()

# %% Row-level plots

categorical_columns = [
    "variant_type",
    "zygosity",
    "variant_inheritance",
    "engineered_cells",
    "patient_cells_tissues",
    "animal_model",
    "in_supplement",
]

for col in categorical_columns:
    normalized_value_counts = df[col].groupby(df["group"]).value_counts(normalize=True)
    normalized_value_counts = normalized_value_counts.rename("proportion").reset_index()
    mplot = sns.barplot(data=normalized_value_counts, x=col, y="proportion", hue="group")
    mplot.set_xticklabels(mplot.get_xticklabels(), rotation=90)
    plt.title(col)
    plt.show()

# %%
