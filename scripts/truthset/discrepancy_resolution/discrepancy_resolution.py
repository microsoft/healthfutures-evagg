"""Truthset Discrepancy Resolution.

This script reads the discrepancy resolution data from Excel files provided by the analysts and compares the
responses to identify discrepancies between the analysts. The discrepancies are saved to separate files for further
analysis.
"""

import argparse

# Libraries
import os

import pandas as pd
from pandas import DataFrame


def parse_discrepancy_excel(file_path: str) -> DataFrame:
    """This function parses the discrepancy resolution Excel file and returns a DataFrame with the parsed data."""
    df = pd.read_csv(file_path, header=None, encoding="unicode_escape")

    genes = []
    papers = []
    links = []
    question_numbers = []
    questions_themselves = []
    all_responses = []
    all_notes = []
    all_phenotypes = []

    i = 0
    while i < len(df):
        if df.iloc[i, 0] == "Gene":
            gene = df.iloc[i, 1]
            paper = df.iloc[i + 1, 1]
            link = df.iloc[i + 2, 1]
            i += 3  # Move to the next row after the link

            while i < len(df) and df.iloc[i, 0] != "Gene":
                cell_value = str(df.iloc[i, 1]) if pd.notna(df.iloc[i, 1]) else ""
                if cell_value.startswith("Q"):
                    question_number, question_itself = cell_value.split(".", 1)

                    responses = []
                    phenotypes = []
                    optional_note = ""
                    j = i + 1
                    while j < len(df) and pd.notna(df.iloc[j, 2]):
                        if df.iloc[j, 2] == "[Optional] Note:":
                            optional_note = df.iloc[j, 3] if pd.notna(df.iloc[j, 3]) else ""
                            j += 1
                            continue
                        response = df.iloc[j, 2]
                        phenotype = df.iloc[j, 3] if pd.notna(df.iloc[j, 3]) else ""
                        responses.append(response)
                        phenotypes.append(phenotype)
                        j += 1

                    for response, phenotype in zip(responses, phenotypes):
                        genes.append(gene)
                        papers.append(paper)
                        links.append(link)
                        question_numbers.append(question_number)
                        questions_themselves.append(question_itself)
                        all_responses.append(response)
                        all_notes.append(optional_note)
                        all_phenotypes.append(phenotype)

                    i = j  # Move to the next question or gene
                else:
                    i += 1  # Move to the next row
        else:
            i += 1  # Move to the next row if not "Gene"

    # Create a DataFrame with the parsed data
    parsed_df = pd.DataFrame(
        {
            "Gene": genes,
            "Paper": papers,
            "Link": links,
            "Q###": question_numbers,
            "Questions": questions_themselves,
            "Response": all_responses,
            "Phenotype": all_phenotypes,
            "Note": all_notes,
        }
    )

    return parsed_df


def compare_discrepancy_dfs(df1: DataFrame, df2: DataFrame) -> DataFrame:
    """This compares two discrepancy DataFrames and returns a subset of the rows where the responses are different."""
    # Merge the two dataframes on the "Q###", "Gene", "Paper", and "Link" columns
    merged_df = df1.merge(df2, on=["Q###", "Gene", "Paper", "Link"], suffixes=("_1", "_2"))

    # Filter the merged dataframe to include only rows where the responses are different
    filtered_df = merged_df[
        (merged_df["Response_1"] != merged_df["Response_2"])
        | ((merged_df["Response_1"] == "Other (see note)") & (merged_df["Response_2"] == "Other (see note)"))
    ]

    # Create a subset of the filtered dataframe with the required columns
    subset_df = filtered_df[
        [
            "Q###",
            "Gene",
            "Paper",
            "Link",
            "Questions_1",
            "Response_1",
            "Phenotype_1",
            "Note_1",
            "Response_2",
            "Phenotype_2",
            "Note_2",
        ]
    ]

    # Filter out rows where Phenotype_1 and Phenotype_2 do not match
    # TODO: Implement different logic to never consider these rows
    subset_df = subset_df[subset_df["Phenotype_1"] == subset_df["Phenotype_2"]]

    return subset_df.reset_index(drop=True)


def main(args: argparse.Namespace) -> None:

    # Ensure the directory exists
    output_dir = os.path.join(".out", "discrepancy_resolution")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parsing the discrepancy Excel files for each analyst
    parsed_df_analyst1 = parse_discrepancy_excel(args.analyst_1_file)
    parsed_df_analyst2 = parse_discrepancy_excel(args.analyst_2_file)
    parsed_df_analyst3 = parse_discrepancy_excel(args.analyst_3_file)

    # Compare the discrepancy DataFrames for each pair of analysts
    subset_df_1_2 = compare_discrepancy_dfs(parsed_df_analyst1, parsed_df_analyst2)
    subset_df_1_3 = compare_discrepancy_dfs(parsed_df_analyst1, parsed_df_analyst3)
    subset_df_2_3 = compare_discrepancy_dfs(parsed_df_analyst2, parsed_df_analyst3)

    # Save the discrepancies to .out files
    subset_df_1_2.to_csv(os.path.join(output_dir, "discrepancies_analyst1_vs_analyst2.tsv"), index=False, sep="\t")
    subset_df_1_3.to_csv(os.path.join(output_dir, "discrepancies_analyst1_vs_analyst3.tsv"), index=False, sep="\t")
    subset_df_2_3.to_csv(os.path.join(output_dir, "discrepancies_analyst2_vs_analyst3.tsv"), index=False, sep="\t")

    # Output the number of discordances between each pair of analysts
    print(
        f"Discordances between Analyst 1 and Analyst 2 on {subset_df_1_2['Q###'].nunique()} questions, "
        f"with {len(subset_df_1_2)} different responses."
    )

    print(
        f"Discordances between Analyst 1 and Analyst 3 on {subset_df_1_3['Q###'].nunique()} questions, "
        f"with {len(subset_df_1_3)} different responses."
    )

    print(
        f"Discordances between Analyst 2 and Analyst 3 on {subset_df_2_3['Q###'].nunique()} questions, "
        f"with {len(subset_df_2_3)} different responses."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Discrepancy Resolution.")
    parser.add_argument(
        "--analyst-1-file",
        type=str,
        default=(r"data/discrepancy_resolution/ana1_discrepancy_resolution.csv"),
        help=("data/discrepancy_resolution/ana1_discrepancy_resolution.csv"),
    )
    parser.add_argument(
        "--analyst-2-file",
        type=str,
        default=(r"data/discrepancy_resolution/ana2_discrepancy_resolution.csv"),
        help=("data/discrepancy_resolution/ana2_discrepancy_resolution.csv"),
    )
    parser.add_argument(
        "--analyst-3-file",
        type=str,
        default=(r"data/discrepancy_resolution/ana3_discrepancy_resolution.csv"),
        help=("data/discrepancy_resolution/ana3_discrepancy_resolution.csv"),
    )

    args = parser.parse_args()

    main(args)
