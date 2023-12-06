# %% Imports.
from typing import Any, Dict

import pandas as pd

from lib.evagg import TruthsetFileLibrary
from lib.evagg.types import Paper, Query

# %% Constants.
truth_table_path = "/home/azureuser/repos/ev-agg-exp/.data/truth/truth_set_small.tsv"

# %% Obtain the truth table.

truth_raw = pd.read_csv(truth_table_path, sep="\t")
truth_papers = truth_raw[["query_gene", "doi", "pmid", "pmcid", "is_pmc_oa"]].drop_duplicates()

# replace nan values for is_pmc_oa with False and change dtype to bool
truth_papers["is_pmc_oa"] = truth_papers["is_pmc_oa"].fillna(False).astype(bool)
truth_papers["pmcid"] = truth_papers["pmcid"].fillna("").astype(str)
truth_papers["pmid"] = truth_papers["pmid"].astype(str)

# %% For the list of queries (genes) in the truth table, attempt to find papers.

# This should change to whatever FileLibrary we want to evaluate.
truthset_library = TruthsetFileLibrary(truth_table_path)


def paper_to_dict(paper: Paper) -> Dict[str, Any]:
    return {
        "doi": paper.id,
        "pmid": paper.props["pmid"],
        "pmcid": paper.props["pmcid"],
        "is_pmc_oa": paper.props["is_pmc_oa"],
    }


def process_group(group_df: pd.DataFrame) -> pd.DataFrame:
    if len(group_df) == 0:
        return group_df

    gene = group_df["query_gene"].iloc[0]

    papers = truthset_library.search(Query(variant=f"{gene}:*"))

    found = pd.DataFrame.from_records([paper_to_dict(paper) for paper in papers])
    found["query_gene"] = gene

    return pd.merge(
        group_df,
        found,
        on=["query_gene", "doi"],
        how="outer",
        suffixes=[None, "_found"],
        indicator=True,
    )


result = truth_papers.groupby("query_gene").apply(process_group)

# quick sanity check
for column_root in ["pmid", "pmcid", "is_pmc_oa"]:
    if not result[column_root].equals(result[f"{column_root}_found"]):
        print(result[[column_root, f"{column_root}_found"]])
        raise ValueError(f"Column {column_root} does not match {column_root}_found")


result.drop(columns=[c for c in result.columns if c.endswith("_found")], inplace=True)

result["in_truthset"] = (result["_merge"] == "both") | (result["_merge"] == "left_only")
result["in_found"] = (result["_merge"] == "both") | (result["_merge"] == "right_only")

result.drop(columns=["_merge"], inplace=True)

# %% Compare the results to the truth table.

# Coarse recall analysis
print(f"Coarse recall: {result.loc[result['in_truthset']==True]['in_found'].mean()}")
print(
    f"  Of {len(result.loc[result['in_truthset']==True])} papers in truthset, {result.loc[result['in_truthset']==True]['in_found'].sum()} were found."
)

# Coarse precision analysis
print(f"Coarse precision: {result.loc[result['in_found']==True]['in_truthset'].mean()}")
print(
    f" Of {len(result.loc[result['in_found']==True])} papers found, {result.loc[result['in_found']==True]['in_truthset'].sum()} were in truthset."
)

# %% Full results

print()
print(result)
