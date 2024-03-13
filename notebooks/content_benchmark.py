"""This script compares two evagg output tables, specifically focusing on the content extraction performance.

Content extraction has two logical components:
1. Identifying the variants in the paper.
2. Extracting the content associated with those variants.

This notebook compares the performance of the two components separately.
"""

# %% Imports.

import os

import pandas as pd

# %% Constants.

TRUTH_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "v1", "evidence_train_v1.tsv")
TEST_PATH = os.path.join(os.path.dirname(__file__), "..", ".out", "content_benchmark.tsv")

# TODO: after we rethink variant nomenclature, figure out whether we need to check the hgvs nomenclatures for agreement.
# CONTENT_COLUMNS = {"phenotype", "zygosity", "variant_inheritance"}
CONTENT_COLUMNS = {"variant_inheritance"}
INDEX_COLUMNS = {"gene", "hgvs_c", "paper_id"}

# %% Read in the truth and test tables.

truth_df = pd.read_csv(TRUTH_PATH, sep="\t")
if "doi" in truth_df.columns:
    print("Warning: converting doi to paper_id")
    truth_df.rename(columns={"doi": "paper_id"}, inplace=True)

test_df = pd.read_csv(TEST_PATH, sep="\t", skiprows=1)

# TODO: temporary, sample the both dfs so we have some missing/extra rows.
# truth_df = truth_df.sample(frac=0.9, replace=False)
# test_df = test_df.sample(frac=0.7, replace=False)

# %% Sanity check the dataframes.

missing_from_truth = CONTENT_COLUMNS.union(INDEX_COLUMNS) - set(truth_df.columns)
if missing_from_truth:
    raise ValueError(f"Truth table is missing columns: {missing_from_truth}")

missing_from_test = CONTENT_COLUMNS.union(INDEX_COLUMNS) - set(test_df.columns)
if missing_from_test:
    raise ValueError(f"Test table is missing columns: {missing_from_test}")

# Ensure that the index columns are unique.
if not truth_df.set_index(list(INDEX_COLUMNS)).index.is_unique:
    raise ValueError("Truth table has non-unique index columns.")

if not test_df.set_index(list(INDEX_COLUMNS)).index.is_unique:
    raise ValueError("Test table has non-unique index columns.")

# %% Merge the dataframes.
columns_of_interest = list(INDEX_COLUMNS.union(CONTENT_COLUMNS))
truth_df = truth_df[columns_of_interest]
test_df = test_df[columns_of_interest]

# Add a column for provenance.
truth_df["in_truth"] = True
test_df["in_test"] = True

# reindex the two dataframes based on a multi_index from INDEX_COLUMNS.
truth_df.set_index(list(INDEX_COLUMNS), inplace=True)
test_df.set_index(list(INDEX_COLUMNS), inplace=True)

# Merge the two dataframes.
merged_df = pd.merge(truth_df, test_df, how="outer", left_index=True, right_index=True, suffixes=["_truth", "_test"])

merged_df.in_test.fillna(False, inplace=True)
merged_df.in_truth.fillna(False, inplace=True)

# %% Assess variant finding.

precision = merged_df.in_truth[merged_df.in_test == True].mean()
recall = merged_df.in_test[merged_df.in_truth == True].mean()

print("---- Variant finding performance ----")
print(f"Variant finding precision: {precision}")
print(f"Variant finding recall: {recall}")
print()

# %% Assess content extraction.

shared_df = merged_df[merged_df.in_truth & merged_df.in_test]

print("---- Content extraction performance ----")

for column in CONTENT_COLUMNS:
    match = shared_df[f"{column}_truth"].str.lower() == shared_df[f"{column}_test"].str.lower()
    print(f"Content extraction accuracy for {column}: {match.mean()}")

    for idx, row in shared_df[~match].iterrows():
        print(f"  Mismatch ({idx}): {row[f'{column}_truth']} != {row[f'{column}_test']}")
    print()

# %%
