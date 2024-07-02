# %% Imports.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %% Constants.

PIPELINE_TSV = ".out/run_evagg_pipeline_20240701_025327/pipeline_benchmark.tsv"

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
pivot_df = pivot_df[pivot_df.sum(axis=1) >= 10]

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

# # group df by gene and get the first row for each gene

# import os

# elts = os.listdir(".out/run_evagg_pipeline_20240701_213144/results_cache/PromptBasedContentExtractor")

# for elt in elts:
#     if any(pmid in elt for pmid in pmids):
#         print(f"deleting {elt}")
#         #os.remove(f".out/run_evagg_pipeline_20240701_213144/results_cache/PromptBasedContentExtractor/{elt}")

# %% Explore individual identifiers.

# On a per paper basis, assess whether a given paper has any individual_ids that are just numeric or whether that paper has individual_ids that are a substring of any other individual ids.
for group in df.groupby("paper_id"):
    paper_id, paper_df = group
    individual_ids = paper_df["individual_id"].unique()
    numeric_individual_ids = [individual_id for individual_id in individual_ids if individual_id.isnumeric()]
    if numeric_individual_ids:
        print(f"Paper {paper_id} has numeric individual ids: {numeric_individual_ids}")
    for individual_id in individual_ids:
        if any(
            individual_id in other_individual_id
            for other_individual_id in individual_ids
            if individual_id != other_individual_id
        ):
            print(
                f"Paper {paper_id} has individual id {individual_id} that is a substring of another individual id. {[other for other in individual_ids if individual_id in other and individual_id != other]}"
            )

# %%
