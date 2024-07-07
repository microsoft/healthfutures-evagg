# %% Imports.

from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %% Constants.

PIPELINE_TSV = ".out/run_evagg_pipeline_20240703_194329/pipeline_benchmark.tsv"

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

# %% Print a handful o links/papers to investigate:

pmids = pivot_df.index[:20]
keep = df[df["paper_id"].isin(pmids)]
for grp_id, grp_df in keep.groupby("paper_id"):
    print(f"{grp_id} - {grp_df['gene'].iloc[0]} - {grp_df['link'].iloc[0]}")

# %% Determine the total number of validation errors for the top 20 papers.

# Sum all the columns in pivot_df except for 'OK'
pivot_df["total_errors"] = pivot_df.drop(columns="OK").sum(axis=1)

# Make a line plot showing the cumulative sum of validation errors as a function of number of papers considered.
plt.figure(figsize=(18, 6))

plt.plot(pivot_df["total_errors"].cumsum() / pivot_df["total_errors"].sum(), marker="o")

# Add a vertical line at 20 papers
plt.axvline(20, color="red", linestyle="--")

# %%

# # group df by gene and get the first row for each gene

# import os

# elts = os.listdir(".out/run_evagg_pipeline_20240701_213144/results_cache/PromptBasedContentExtractor")

# for elt in elts:
#     if any(pmid in elt for pmid in pmids):
#         print(f"deleting {elt}")
#         #os.remove(f".out/run_evagg_pipeline_20240701_213144/results_cache/PromptBasedContentExtractor/{elt}")

# %% Explore individual identifiers.

# Do a quick scan for suspicious things.
for group in df.groupby("paper_id"):
    paper_id, paper_df = group
    individual_ids = paper_df["individual_id"].unique()
    numeric_individual_ids = [individual_id for individual_id in individual_ids if individual_id.isnumeric()]
    # if numeric_individual_ids:
    #     print(f"Paper {paper_id} has numeric individual ids: {numeric_individual_ids}")
    for individual_id in individual_ids:
        if any(
            individual_id in other_individual_id
            for other_individual_id in individual_ids
            if individual_id != other_individual_id
        ):
            print(
                f"Paper {paper_id} has individual id {individual_id} that is a substring of another individual id. {[other for other in individual_ids if individual_id in other and individual_id != other]}"
            )

# %% Evaluate QC deduping of individual IDs

df_play = df.copy()

for group in df_play.groupby("paper_id"):
    paper_id, paper_df = group
    individual_ids = paper_df["individual_id"].unique()
    tokenized = {
        individual_id: individual_id.replace(".", "").replace("-", "").replace(":", "").split(" ")
        for individual_id in individual_ids
    }

    for key, key_tokens in tokenized.items():
        for other_key, other_key_tokens in tokenized.items():
            if key != other_key and all(token in other_key_tokens for token in key_tokens):
                print(f"Paper {paper_id} has individual id {key} that is similar to {other_key}")
                print(
                    f"  Potentially {(paper_df.individual_id == key).sum()} and {(paper_df.individual_id == other_key).sum()} rows affected"
                )


# %% Count the number of rows associated with individual IDs that end in 's'

counts = defaultdict(int)

for id, row in df.iterrows():
    if row.individual_id.endswith("s"):
        # print(row.individual_id)
        counts[row.individual_id] += 1

for key, value in counts.items():
    print(f"{key}: {value}")

# %% Examine the ratio of unique values for "paper_variant" to the total number of rows for each paper.

# Group by paper_id and count the number of unique paper_variants
unique_variants = df.groupby("paper_id")["paper_variant"].nunique()

# Count the total number of rows for each paper
total_rows = df.groupby("paper_id").size()

# Calculate the ratio of unique paper_variants to total rows
ratios = unique_variants / total_rows

# Sort the ratios
ratios = ratios.sort_values(ascending=False)

# Print the 20 papers with the lowest ratios.
for paper_id, ratio in ratios.tail(20).items():
    print(
        f"{paper_id}: {ratio} [{df[df.paper_id == paper_id].iloc[0].gene}] [{df[df.paper_id == paper_id].iloc[0].link}]"
    )

# Plot the ratios
plt.figure(figsize=(18, 6))
ratios.tail(50).plot(kind="bar")
plt.xlabel("pmid")
plt.ylabel("Unique variants / Total rows")
plt.title("Ratio of unique variants to total rows by pmid")
plt.show()

# %% Examine the number of variants for each individual ID (not including "unknown")

df_no_unknown = df[df.individual_id != "unknown"]

for paper_id, paper_df in df_no_unknown.groupby("paper_id"):
    for individual_id, individual_df in paper_df.groupby("individual_id"):
        if individual_df.shape[0] >= 5:
            print(
                f"{paper_id} / {individual_id} ({individual_df.shape[0]}) [{individual_df.iloc[0].gene}] [{individual_df.iloc[0].link}]"
            )


# %%
