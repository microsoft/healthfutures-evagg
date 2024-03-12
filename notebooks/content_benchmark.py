"""This script compares two evagg output tables, specifically focusing on the content extraction performance.

Content extraction has two logical components:
1. Identifying the observations (Tuple[variant, individual]) in the paper.
2. Extracting the content associated with those observations.

This notebook compares the performance of the two components separately.
"""

# %% Imports.

import os
from typing import Set

import pandas as pd

# %% Constants.

TRUTH_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "v1", "evidence_train_v1.tsv")
TEST_PATH = os.path.join(os.path.dirname(__file__), "..", ".out", "observation_benchmark.tsv")

# TODO: after we rethink variant nomenclature, figure out whether we need to check the hgvs nomenclatures for agreement.
CONTENT_COLUMNS: Set[str] = set()  # when CONTENT_COLUMNS is empty we're just comparing observation-finding
# CONTENT_COLUMNS = {"phenotype", "zygosity", "variant_inheritance"} # noqa
INDEX_COLUMNS = {"individual_id", "hgvs_c", "hgvs_p", "paper_id"}
EXTRA_COLUMNS = {"gene", "in_supplement"}

RESTRICT_TRUTH_GENES_TO_TEST = True  # if True, only compare the genes in the test set to the truth set.

# %% Read in the truth and test tables.

truth_df = pd.read_csv(TRUTH_PATH, sep="\t")
if "doi" in truth_df.columns:
    print("Warning: converting doi to paper_id")
    truth_df.rename(columns={"doi": "paper_id"}, inplace=True)

test_df = pd.read_csv(TEST_PATH, sep="\t", skiprows=1)

# %% Restrict the truth set to the genes in the test set.
if RESTRICT_TRUTH_GENES_TO_TEST:
    print("Warning: restricting truth set to genes in the test set.")
    test_genes = set(test_df.gene.unique())
    truth_df = truth_df[truth_df.gene.isin(test_genes)]

# TODO: temporary, sample the both dfs so we have some missing/extra rows.
# truth_df = truth_df.sample(frac=0.9, replace=False)
# test_df = test_df.sample(frac=0.7, replace=False)

# %% Sanity check the dataframes.
all_columns = CONTENT_COLUMNS.union(INDEX_COLUMNS).union(EXTRA_COLUMNS)

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

# %% Consolidate the indices.

if "hgvs_c" in INDEX_COLUMNS and "hgvs_p" in INDEX_COLUMNS:
    print("Refactoring index columns to use hgvs_desc.")

    # Some variants are described with hgvs_c (splicing), some with hgvs_p (protein variant), and some can have both.
    # We want to consolidate these into a single index column, with preference to hgvs_c if it's not nan.
    truth_df["hgvs_desc"] = truth_df["hgvs_c"].fillna(truth_df["hgvs_p"])
    test_df["hgvs_desc"] = test_df["hgvs_c"].fillna(test_df["hgvs_p"])

    # remove the original hgvs colummns from INDEX_COLUMNS and add the new index
    INDEX_COLUMNS -= {"hgvs_c", "hgvs_p"}
    INDEX_COLUMNS.add("hgvs_desc")

    # Reset all_columns
    all_columns = CONTENT_COLUMNS.union(INDEX_COLUMNS).union(EXTRA_COLUMNS)


# %% Merge the dataframes.
columns_of_interest = list(all_columns)

truth_df = truth_df.reset_index()
truth_df = truth_df.reset_index()[[c for c in columns_of_interest if c in truth_df.columns]]

test_df = test_df.reset_index()
test_df = test_df[[c for c in columns_of_interest if c in test_df.columns]]


# Add a column for provenance.
truth_df["in_truth"] = True
truth_df["in_truth"] = truth_df["in_truth"].astype("boolean")  # support nullable.
test_df["in_test"] = True
test_df["in_test"] = test_df["in_test"].astype("boolean")  # support nullable.

# reindex the two dataframes based on a multi_index from INDEX_COLUMNS.
truth_df.set_index(list(INDEX_COLUMNS), inplace=True)
test_df.set_index(list(INDEX_COLUMNS), inplace=True)

# Merge the two dataframes.
merged_df = pd.merge(truth_df, test_df, how="outer", left_index=True, right_index=True, suffixes=["_truth", "_test"])

merged_df["in_truth"] = merged_df["in_truth"].fillna(False)
merged_df["in_test"] = merged_df["in_test"].fillna(False)

if "gene_truth" in merged_df.columns:
    merged_df["gene"] = merged_df["gene_truth"].fillna(merged_df["gene_test"])
    merged_df.drop(columns=["gene_truth", "gene_test"], inplace=True)

# Reorder columns, keeping in_truth and in_test as the last two.
merged_df = merged_df[[c for c in merged_df.columns if c not in {"in_truth", "in_test"}] + ["in_truth", "in_test"]]

# %% Assess variant finding.

precision = merged_df.in_truth[merged_df.in_test == True].mean()
recall = merged_df.in_test[merged_df.in_truth == True].mean()

# Make a copy of merged_df removing all rows where in_supplement is 'Y'
merged_df_no_supplement = merged_df[merged_df.in_supplement != "Y"]
precision_no_supplement = merged_df_no_supplement.in_truth[merged_df_no_supplement.in_test == True].mean()
recall_no_supplement = merged_df_no_supplement.in_test[merged_df_no_supplement.in_truth == True].mean()

print("---- Variant finding performance ----")
print("Overall")
print(f"  Variant finding precision: {precision:.2f}")
print(f"  Variant finding recall: {recall:.2f}")
print("Ignoring truth papers from supplement")
print(f"  Variant finding precision: {precision_no_supplement:.2f}")
print(f"  Variant finding recall: {recall_no_supplement:.2f}")
print()

print(merged_df)

# %% Assess content extraction.

if CONTENT_COLUMNS:
    shared_df = merged_df[merged_df.in_truth & merged_df.in_test]

    print("---- Content extraction performance ----")

    for column in CONTENT_COLUMNS:
        match = shared_df[f"{column}_truth"].str.lower() == shared_df[f"{column}_test"].str.lower()
        print(f"Content extraction accuracy for {column}: {match.mean()}")

        for idx, row in shared_df[~match].iterrows():
            print(f"  Mismatch ({idx}): {row[f'{column}_truth']} != {row[f'{column}_test']}")
        print()


# %%
