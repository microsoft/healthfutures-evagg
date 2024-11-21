"""
The objective of this script is to extract summary statistics about the manual ground truth data.
"""

import argparse
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd


def read_data(file_path: str, sep: str = "\t") -> pd.DataFrame:
    """Reads a file"""
    return pd.read_csv(file_path, sep=sep)


def combine_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Combines two dataframes"""
    return pd.concat([df1, df2])


def calculate_papers_per_gene(df: pd.DataFrame) -> pd.Series:
    """Calculates the number of papers per gene"""
    return df.groupby("gene").size()  # type: ignore


def calculate_accessible_papers_per_gene(df: pd.DataFrame) -> pd.Series:
    """Calculates the number of accessible papers per gene"""
    return df[df["can_access"] == True].groupby("gene").size()  # type: ignore


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
    df_train = read_data(file_path_train)
    df_test = read_data(file_path_test)
    combined_df = combine_dataframes(df_train[["gene", "evidence_base"]], df_test[["gene", "evidence_base"]])
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
    submission_df = read_data(submission_file_path)
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
    group_assignments = read_data(group_assignments_file_path)
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
    df = read_data(file_path)
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


def main(args) -> None:

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.outdir, f"mgt_table_stats_{timestamp}/")
    os.makedirs(output_dir, exist_ok=True)

    # File paths
    file_path_train = "data/v1/papers_train_v1.tsv"
    file_path_test = "data/v1/papers_test_v1.tsv"
    evidence_train_file_path = "data/v1/evidence_train_v1.tsv"
    evidence_test_file_path = "data/v1/evidence_test_v1.tsv"
    submission_file_path = "data/gencc_clingen_submit_date.tsv"
    group_assignments_file_path = "data/v1/group_assignments.tsv"

    # Read and combine data
    df_train = read_data(file_path_train)
    df_test = read_data(file_path_test)
    df_combined = combine_dataframes(df_train, df_test)

    # Calculate papers per gene
    papers_per_gene = calculate_papers_per_gene(df_combined)
    accessible_papers_per_gene = calculate_accessible_papers_per_gene(df_combined)

    # Create result DataFrame
    result = create_result_dataframe(papers_per_gene, accessible_papers_per_gene)

    # Identify evidence base
    unique_combined_gene_evidence_base_df = identify_evidence_base(evidence_train_file_path, evidence_test_file_path)

    # Merge dataframes
    gene_to_evidence_base = {
        "RHOH": "L",
        "CPT1B": "N",
        "TAPBP": "M",
        "HYAL1": "M",
        "RGS9": "M",
        "PTCD3": "L",
        "IGKC": "L",
        "PEX11G": "N",
        "OTUD7A": "L",
        "MPST": "N",
        "KMO": "N",
        "ADCY1": "L",
    }
    evidence_base_mapping = {"L": "Limited", "N": "No Known", "M": "Moderate"}
    merged_result = merge_dataframes(
        result, unique_combined_gene_evidence_base_df, gene_to_evidence_base, evidence_base_mapping
    )

    # Add submission data
    merged_result_w_gencc = add_submission_data(merged_result, submission_file_path)

    # Split train and test
    train_df, test_df = split_train_test(merged_result_w_gencc, group_assignments_file_path)

    # Save the train and test dataframes to a file
    train_df.to_csv(output_dir + "train_paper_summary.tsv", sep="\t", index=False)
    test_df.to_csv(output_dir + "test_paper_summary.tsv", sep="\t", index=False)

    # Calculate grouped means
    train_grouped = calculate_grouped_means(train_df)
    print("\nTraining Set Average Statistics per Evidence Base")
    print(train_grouped)
    test_grouped = calculate_grouped_means(test_df)
    print("\nTest Set Average Statistics per Evidence Base")
    print(test_grouped)

    # Save the training and test set average stats per evidence base
    train_grouped.to_csv(output_dir + "train_avg_stats.tsv", sep="\t", index=False)
    test_grouped.to_csv(output_dir + "test_avg_stats.tsv", sep="\t", index=False)

    # Extract stats from train and test data
    summaries_train, average_patients_train, unique_papers_train, patients_per_paper_dict_train = extract_stats(
        evidence_train_file_path
    )
    summaries_test, average_patients_test, unique_papers_test, patients_per_paper_dict_test = extract_stats(
        evidence_test_file_path
    )

    # Print combined summary table
    print(
        "\nManual Ground Truth statistics. M: moderate evidence type, L: limited evidence type, N: no known gene-disease relationship type"
    )
    create_large_summary_table(
        summaries_train,
        summaries_test,
        average_patients_train,
        average_patients_test,
        unique_papers_train,
        unique_papers_test,
        patients_per_paper_dict_train,
        patients_per_paper_dict_test,
        output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Ground Truth (MGT) Table Statistics")
    parser.add_argument("--outdir", type=str, default=".out/", help="Output directory")
    args = parser.parse_args()
    main(args)
