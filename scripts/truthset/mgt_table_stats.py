"""
The objective of this script is to extract summary statistics about the manual ground truth data.
"""

# %%

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats as stats
import yaml

# %% Function definitions.


def create_result_dataframe(papers_per_gene: pd.Series, accessible_papers_per_gene: pd.Series) -> pd.DataFrame:
    """Creates a DataFrame of total papers, accessible papers, and proportion accessible papers per gene"""
    result = (
        pd.DataFrame({"total_papers": papers_per_gene, "accessible_papers": accessible_papers_per_gene})
        .fillna(0)
        .astype(int)
    )
    result["proportion_accessible"] = (result["accessible_papers"] / result["total_papers"] * 100).round(0).astype(int)
    result = (
        result.drop(columns=["accessible_papers"])
        .sort_values(by="proportion_accessible", ascending=False)
        .reset_index()
    )
    return result


def identify_evidence_base(file_path_train: str, file_path_test: str) -> pd.DataFrame:
    """Identifies the evidence base for each gene and combines the train and test data"""
    df_train = pd.read_csv(file_path_train, sep="\t")
    df_test = pd.read_csv(file_path_test, sep="\t")
    combined_df = pd.concat([df_train[["gene", "evidence_base"]], df_test[["gene", "evidence_base"]]])
    return combined_df.drop_duplicates().reset_index(drop=True)


def merge_dataframes(
    result: pd.DataFrame,
    unique_combined_gene_evidence_base_df: pd.DataFrame,
    gene_to_evidence_base: dict,
    evidence_base_mapping: dict,
) -> pd.DataFrame:
    """Merges the result DataFrame with the evidence base DataFrame to form a DataFrame of genes, evidence base, total papers, and proportion accessible papers"""
    merged_result = pd.merge(result, unique_combined_gene_evidence_base_df, on="gene", how="left")
    merged_result["evidence_base"] = (
        merged_result["gene"].map(gene_to_evidence_base).fillna(merged_result["evidence_base"])
    )
    merged_result["evidence_base"] = merged_result["evidence_base"].replace(evidence_base_mapping)
    merged_result["evidence_base"] = merged_result["evidence_base"].replace("No Known Disease Relationship", "No Known")
    sort_order = pd.CategoricalDtype(["Moderate", "Limited", "No Known"], ordered=True)
    merged_result["evidence_base"] = merged_result["evidence_base"].astype(sort_order)
    merged_result = merged_result.sort_values(
        by=["evidence_base", "proportion_accessible"], ascending=[True, False]
    ).reset_index(drop=True)
    return merged_result[["gene", "evidence_base", "total_papers", "proportion_accessible"]]


def add_submission_data(merged_result: pd.DataFrame, submission_file_path: str) -> pd.DataFrame:
    """Adds GenCC ClinGen submission and # PubMed Papers to the merged result DataFrame"""
    submission_df = pd.read_csv(submission_file_path, sep="\t")
    submission_df = submission_df.rename(columns={"submitted_as_classification_name": "gene"})
    merged_result_w_gencc = merged_result.merge(
        submission_df[["gene", "Date_of_first_GenCC_ClinGen_submission", "#_returned_PubMed_papers"]],
        on="gene",
        how="left",
    )
    return merged_result_w_gencc.rename(
        columns={
            "Date_of_first_GenCC_ClinGen_submission": "GenCC ClinGen submission",
            "#_returned_PubMed_papers": "# PubMed Papers",
        }
    )


def split_train_test(
    merged_result_w_gencc: pd.DataFrame, group_assignments_file_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the merged result DataFrame into train and test DataFrames"""
    group_assignments = pd.read_csv(group_assignments_file_path, sep="\t")
    merged_result_w_gencc = merged_result_w_gencc.merge(group_assignments, on="gene", how="left")
    train_df = (
        merged_result_w_gencc[merged_result_w_gencc["group"] == "train"].drop(columns=["group"]).reset_index(drop=True)
    )
    test_df = (
        merged_result_w_gencc[merged_result_w_gencc["group"] == "test"].drop(columns=["group"]).reset_index(drop=True)
    )
    return train_df, test_df


def calculate_grouped_means(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the grouped means of total papers, proportion accessible papers, and # PubMed Papers"""
    return (
        df.groupby("evidence_base")
        .agg({"total_papers": "mean", "proportion_accessible": "mean", "# PubMed Papers": "mean"})
        .reset_index()
    )


def extract_stats(file_path: str) -> tuple[dict, dict, dict, dict]:
    """Extracts summary statistics from the evidence file"""
    df = pd.read_csv(file_path, sep="\t")

    df["individual_id"] = df["individual_id"].replace(to_replace=re.compile(r"unknown", re.IGNORECASE), value=np.nan)  # type: ignore
    evidence_base_groups = df["evidence_base"].unique()
    columns_to_summarize = [
        "paper_build",
        "variant_type",
        "zygosity",
        "variant_inheritance",
        "study_type",
        "engineered_cells",
        "patient_cells_tissues",
        "animal_model",
        "can_access",
        "license",
        "individual_id",
    ]
    summaries, average_patients, unique_papers, patients_per_paper_dict = {}, {}, {}, {}

    for evidence_base in evidence_base_groups:
        group_df = df[df["evidence_base"] == evidence_base]
        summaries[evidence_base] = {column: group_df[column].value_counts() for column in columns_to_summarize}
        patients_per_paper = group_df.groupby("paper_id")["individual_id"].apply(lambda x: x.dropna().count()).dropna()
        patients_per_paper_dict[evidence_base] = patients_per_paper
        average_patients[evidence_base] = {
            "mean": round(patients_per_paper.mean(), 2),
            "median": round(patients_per_paper.median(), 2),
            "mode": patients_per_paper.mode(),
        }
        unique_papers[evidence_base] = group_df["paper_id"].nunique()

    return summaries, average_patients, unique_papers, patients_per_paper_dict


def create_large_summary_table(
    summaries_train: dict,
    summaries_test: dict,
    average_patients_train: dict,
    average_patients_test: dict,
    unique_papers_train: dict,
    unique_papers_test: dict,
    patients_per_paper_dict_train: dict,
    patients_per_paper_dict_test: dict,
    output_dir: str,
) -> None:
    """Prints a summary table of the extracted statistics for the train and test DataFrames."""
    headers = ["Category", "Train M", "Test M", "Train L", "Test L", "Train N", "Test N"]
    rows = []

    categories = [
        "paper_build",
        "variant_type",
        "zygosity",
        "variant_inheritance",
        "study_type",
        "engineered_cells",
        "patient_cells_tissues",
        "animal_model",
        "can_access",
        "license",
    ]

    for category in categories:
        for value in set(summaries_train.get("Moderate", {}).get(category, {}).keys()).union(
            summaries_test.get("Moderate", {})
            .get(category, {})
            .keys()  # get all possible values for each column/category
        ):
            row = [f"{category}: {value}"]
            for evidence_base in ["Moderate", "Limited", "No Known Disease Relationship"]:
                row.append(summaries_train.get(evidence_base, {}).get(category, {}).get(value, 0))
                row.append(summaries_test.get(evidence_base, {}).get(category, {}).get(value, 0))
            rows.append(row)

    rows.append(
        [
            "patients per paper: mean",
            average_patients_train.get("Moderate", {}).get("mean", 0),
            average_patients_test.get("Moderate", {}).get("mean", 0),
            average_patients_train.get("Limited", {}).get("mean", 0),
            average_patients_test.get("Limited", {}).get("mean", 0),
            average_patients_train.get("No Known Disease Relationship", {}).get("mean", 0),
            average_patients_test.get("No Known Disease Relationship", {}).get("mean", 0),
        ]
    )
    rows.append(
        [
            "patients per paper: median",
            average_patients_train.get("Moderate", {}).get("median", 0),
            average_patients_test.get("Moderate", {}).get("median", 0),
            average_patients_train.get("Limited", {}).get("median", 0),
            average_patients_test.get("Limited", {}).get("median", 0),
            average_patients_train.get("No Known Disease Relationship", {}).get("median", 0),
            average_patients_test.get("No Known Disease Relationship", {}).get("median", 0),
        ]
    )
    rows.append(
        [
            "patients per paper: mode",
            average_patients_train.get("Moderate", {}).get("mode", [0])[0],
            average_patients_test.get("Moderate", {}).get("mode", [0])[0],
            average_patients_train.get("Limited", {}).get("mode", [0])[0],
            average_patients_test.get("Limited", {}).get("mode", [0])[0],
            average_patients_train.get("No Known Disease Relationship", {}).get("mode", [0])[0],
            average_patients_test.get("No Known Disease Relationship", {}).get("mode", [0])[0],
        ]
    )
    rows.append(
        [
            "# unique papers",
            unique_papers_train.get("Moderate", 0),
            unique_papers_test.get("Moderate", 0),
            unique_papers_train.get("Limited", 0),
            unique_papers_test.get("Limited", 0),
            unique_papers_train.get("No Known Disease Relationship", 0),
            unique_papers_test.get("No Known Disease Relationship", 0),
        ]
    )

    # Create a DataFrame from the rows
    df = pd.DataFrame(rows, columns=headers)

    # Save the DataFrame to a TSV file
    df.to_csv(output_dir + "mgt_category_stats.tsv", sep="\t", index=False)


# def main(args: argparse.Namespace) -> None:
OUTPUT_DIRECTORY = ".out/"

# %%
# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(OUTPUT_DIRECTORY, f"mgt_table_stats_{timestamp}/")
# output_dir = os.path.join(args.outdir, f"mgt_table_stats_{timestamp}/")
os.makedirs(output_dir, exist_ok=True)

# %%
# File paths
papers_train_file_path = "data/v1.1/papers_train_v1.1.tsv"
papers_test_file_path = "data/v1.1/papers_test_v1.1.tsv"

evidence_train_file_path = "data/v1.1/evidence_train_v1.1.tsv"
evidence_test_file_path = "data/v1.1/evidence_test_v1.1.tsv"

submission_file_path = "data/gencc_clingen_submit_date.tsv"
group_assignments_file_path = "data/v1/group_assignments.tsv"

test_config_file_path = "lib/config/queries/mgttest_subset.yaml"
train_config_file_path = "lib/config/queries/mgttrain_subset.yaml"

skipped_pmids_file_path = "scripts/benchmark/paper_finding/paper_finding_benchmarks_skipped_pmids.txt"


# %% Read and combine papers data
train_papers_df = pd.read_csv(papers_train_file_path, sep="\t")
test_papers_df = pd.read_csv(papers_test_file_path, sep="\t")
papers_df = pd.concat([train_papers_df, test_papers_df])

test_config_list = yaml.safe_load(open(test_config_file_path))
train_config_list = yaml.safe_load(open(train_config_file_path))

config_list = test_config_list + train_config_list
config = pd.DataFrame(config_list)

skipped_pmids = pd.read_csv(skipped_pmids_file_path, header=None, names=["pmid"])

evidence_base_df = pd.read_csv(submission_file_path, sep="\t")

# %% Drop all the skipped pmids

skippable = papers_df["pmid"].isin(skipped_pmids["pmid"])
print(f"Skipped {skippable.sum()} papers as they're not available from pubmed using standardized search.")

papers_df = papers_df[~skippable]

# %% Generate the per gene and total papers dataframes

rows = []
for gene, group_df in papers_df.groupby("gene"):
    rows.append(
        {
            "gene": gene,
            "group": group_df["group"].iloc[0],
            "category": evidence_base_df.query("submitted_as_classification_name == @gene")[
                "unique_disease_count"
            ].iloc[0],
            "considered": config.query("gene_symbol == @gene").iloc[0]["retmax"],
            "total": group_df.shape[0],
            "accessible": group_df["can_access"].sum(),
        }
    )

papers_summary_df = pd.DataFrame(rows)

# %% Test for a significant effect of category on the number of papers per gene
# using a kruskall wallace test

stat, p_value = stats.kruskal(
    papers_summary_df.query("category == 'Limited'")["total"],
    papers_summary_df.query("category == 'Moderate'")["total"],
    papers_summary_df.query("category == 'No Known Disease Relationship'")["total"],
)

# %% Print out some interesting facts for the paper.

print(f"Considered papers: {papers_summary_df['considered'].sum()}")
print("  Per gene:")
print(papers_summary_df["considered"].agg(["mean", "std"]).round(2))
print("  Per category:")
print(papers_summary_df.groupby("category")["considered"].agg(["mean", "std"]).round(2))
print("")

print(f"Relevant papers: {papers_summary_df['total'].sum()}")
print("  Per gene:")
print(papers_summary_df["total"].agg(["mean", "std"]).round(2))
print("")

print(f"Accessible papers: {papers_summary_df['accessible'].sum()}")
print("  Per gene:")
print(papers_summary_df["accessible"].agg(["mean", "std"]).round(2))
print("")

# %% Load in the evidence data.

evidence_train_df = pd.read_csv(evidence_train_file_path, sep="\t")
evidence_test_df = pd.read_csv(evidence_test_file_path, sep="\t")
evidence_df = pd.concat([evidence_train_df, evidence_test_df])

evidence_df["hgvs"] = evidence_df["hgvs_c"].fillna(evidence_df["hgvs_p"])

# %% Print out some interesting facts for the evidence.

print(f"Number of observations: {evidence_df.shape[0]}")
print(f"Number of unique variants: {evidence_df.drop_duplicates(["hgvs", "gene"]).shape[0]}")
print(f"Number of unique genes: {evidence_df['gene'].nunique()}")
print(f"Number of unique papers: {evidence_df['paper_id'].nunique()}")

# Generate a new df indexed by gene with the number of observations and unique variants.
gene_evidence_df = (
    evidence_df.groupby("gene")
    .agg({"hgvs": "nunique", "author": "count"})
    .rename({"hgvs": "variants", "author": "observations"}, axis=1)  # author is an arbitrary column choice.
)
paper_evidence_df = (
    evidence_df.groupby(["pmid", "gene"])
    .agg({"hgvs": "nunique", "author": "count"})
    .rename({"hgvs": "variants", "author": "observations"}, axis=1)  # author is an arbitrary column choice.
)

print("Averages by gene:")
print(gene_evidence_df.agg(["mean", "std"]).round(2))
print("")

print("Averages by paper:")
print(paper_evidence_df.agg(["mean", "std"]).round(2))
print("")

# %% Intentionally left blank
