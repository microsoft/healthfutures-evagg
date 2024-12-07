"""This script is used to explore the differences between two versions of the truthset. This is useful for debugging,
though we may not need to keep it around long-term."""

# %% imports.

import pandas as pd

# %% data loading.

group = "train"

v1 = pd.read_csv(f"data/v1/papers_{group}_v1.tsv", sep="\t").set_index("pmid")
v2 = pd.read_csv(f"data/v1.1/papers_{group}_v1.1.tsv", sep="\t").set_index("pmid")

# read all the lines in from scripts/benchmark/paper_finding/paper_finding_benchmarks_skipped_pmids.txt as a list of
# strings.

with open("scripts/benchmark/paper_finding/paper_finding_benchmarks_skipped_pmids.txt") as f:
    skipped_pmids = [int(line.strip()) for line in f.readlines()]

# %% data comparison

# show rows where the pmid exists in only one dataframe.
v1_only = v1[~v1.index.isin(v2.index)]
v2_only = v2[~v2.index.isin(v1.index)]

# join the two dataframes and select rows where the values are different.
joined = v1.join(v2, lsuffix="_v1", rsuffix="_v2", how="inner")
diff = joined[
    (joined["has_fulltext_v1"] != joined["has_fulltext_v2"])
    | (joined["can_access_v1"] != joined["can_access_v2"])
    | (joined["license_v1"] != joined["license_v2"])
]

# %% look at skipped pmids

# list the pmids from v1_only and v2_only that are in skipped_pmids
v1_skipped = v1_only[v1_only.index.isin(skipped_pmids)]
v2_skipped = v2_only[v2_only.index.isin(skipped_pmids)]


# %% print out all the results for easy comparison.

print(f"V1 (N={v1.shape[0]})")
print(f"V2 (N={v2.shape[0]})")
print("\n\n")

print(f"PAPERS IN V1 ONLY (N={v1_only.shape[0]}): ")
print(v1_only)
print("\n\n")

print(f"PAPERS IN V2 ONLY (N={v2_only.shape[0]}): ")
print(v2_only)
print("\n\n")

print(f"DIFFERENCES BETWEEN V1 AND V2 (N={diff.shape[0]}): ")
print(diff)
print("\n\n")

print(f"PAPERS IN V1 THAT WERE SKIPPED (N={v1_skipped.shape[0]}): ")
print(v1_skipped)
print("\n\n")

print(f"PAPERS IN V2 THAT WERE SKIPPED (N={v2_skipped.shape[0]}): ")
print(v2_skipped)
print("\n\n")

# Show the math.
v2_only_n = v2_only.shape[0]
v1_only_n = v1_only.shape[0]

v2_skipped_n = v2_skipped.shape[0]
v1_skipped_n = v1_skipped.shape[0]

print(
    f"New papers for {group}: "
    f"{v2_only_n} (v2 only) - {v1_only_n} (v1 only) + "
    f"{diff.query('can_access_v2 == True').shape[0]} (changed to accessible) + "
    f"{v1_skipped_n} (v1 skipped) - {v2_skipped_n} (v2 skipped) = "
    f"{v2_only_n - v1_only_n + diff.query('can_access_v2 == True').shape[0] + v1_skipped_n - v2_skipped_n}"
)


# %% Intentionally empty.
