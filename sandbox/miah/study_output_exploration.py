# %% Imports.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %% Constants.

PIPELINE_TSV = ".out/pipeline_benchmark.tsv"

# %% Load data.

df = pd.read_csv(PIPELINE_TSV, sep="\t", header=1)

# %% Make a bar plot showing the number of rows for each value of pmid. Color the bars to represent whether the validation_error column is empty or not.

df_small = df[["paper_id", "validation_error"]].copy()
df_small.fillna({"validation_error": "OK"}, inplace=True)

# replace the following values for validation_error with "OTHER"
# ["ECOORDINATESYSTEMMISMATCH", "ENOSELECTORFOUND", "EOFFSET", "EUNCERTAIN"]
df_small.loc[
    df_small["validation_error"].isin(["ECOORDINATESYSTEMMISMATCH", "ENOSELECTORFOUND", "EOFFSET", "EUNCERTAIN"]),
    "validation_error",
] = "OTHER"

grouped = df_small.groupby(["paper_id", "validation_error"]).size().reset_index(name="count")

# %%
# Group by pmid and validation_error, and count the occurrences

# Pivot the DataFrame to have validation_error values as columns
pivot_df = grouped.pivot(index="paper_id", columns="validation_error", values="count").fillna(0)

# Filter pivot_df to only inclode pmids with a total aross all columns of 3 or more.
pivot_df = pivot_df[pivot_df.sum(axis=1) >= 3]

# Sort pivot_df by the total count across all columns
pivot_df = pivot_df.loc[pivot_df.sum(axis=1).sort_values(ascending=False).index]

# Create a bar plot
ax = pivot_df.plot(kind="bar", stacked=True, figsize=(18, 6))
plt.xlabel("pmid")
plt.ylabel("Count")
plt.title("Validation Errors by pmid")
plt.legend(title="Validation Error")
plt.show()
# %%
