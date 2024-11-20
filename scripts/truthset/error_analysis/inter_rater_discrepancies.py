"""Inter-rater Discrepancy Identification

This script reads the discrepancy resolution data from Excel files provided by the analysts and compares the responses
to identify discrepancies between the analysts. The parsed Excel files and the discrepancies are saved to separate
files for further analysis.
"""

import argparse
import os
from datetime import datetime

import pandas as pd
from pandas import DataFrame


def parse_error_analysis_excel(file_path: str) -> DataFrame:
    """This function parses the error analysis Excel file and returns a DataFrame with the parsed data."""
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


def compare_error_analysis_dfs(df1: DataFrame, df2: DataFrame) -> DataFrame:
    """This compares two error analysis DataFrames and returns a subset of the rows where the responses are different."""
    # Merge the two dataframes on the "Q###", "Gene", "Paper", and "Link" columns
    merged_df = df1.merge(df2, on=["Q###", "Gene", "Paper", "Link"], suffixes=("_1", "_2"))

    # Filter the merged df to include only rows where the responses are different, or both have "Other (see note)"
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

    # Generating output directory as needed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.outdir, f"inter_rater_discrep_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Data directory
    data_dir = "data/error_analysis/"

    # Parsing the error analysis Excel files for each analyst
    parsed_df_analyst1 = parse_error_analysis_excel(args.analyst_1_file)
    parsed_df_analyst2 = parse_error_analysis_excel(args.analyst_2_file)
    parsed_df_analyst3 = parse_error_analysis_excel(args.analyst_3_file)

    # Save parsed worksheets to .tsv in data directory as they are used to generate the error analysis summary figures
    parsed_df_analyst1.to_csv(os.path.join(data_dir, "parsed_ana1_error_analysis_worksheet.tsv"), index=False, sep="\t")
    parsed_df_analyst2.to_csv(os.path.join(data_dir, "parsed_ana2_error_analysis_worksheet.tsv"), index=False, sep="\t")
    parsed_df_analyst3.to_csv(os.path.join(data_dir, "parsed_ana3_error_analysis_worksheet.tsv"), index=False, sep="\t")

    # Compare the error_analysis DataFrames for each pair of analysts
    subset_df_1_2 = compare_error_analysis_dfs(parsed_df_analyst1, parsed_df_analyst2)
    subset_df_1_3 = compare_error_analysis_dfs(parsed_df_analyst1, parsed_df_analyst3)
    subset_df_2_3 = compare_error_analysis_dfs(parsed_df_analyst2, parsed_df_analyst3)

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

    parser = argparse.ArgumentParser(description="Inter-rater Discrepancy Identification")
    parser.add_argument(
        "--analyst-1-file",
        type=str,
        default=(r"data/error_analysis/ana1_error_analysis_worksheet.csv"),
        help=("data/error_analysis/ana1_error_analysis_worksheet.csv"),
    )
    parser.add_argument(
        "--analyst-2-file",
        type=str,
        default=(r"data/error_analysis/ana2_error_analysis_worksheet.csv"),
        help=("data/error_analysis/ana2_error_analysis_worksheet.csv"),
    )
    parser.add_argument(
        "--analyst-3-file",
        type=str,
        default=(r"data/error_analysis/ana3_error_analysis_worksheet.csv"),
        help=("data/error_analysis/ana3_error_analysis_worksheet.csv"),
    )
    parser.add_argument("--outdir", default=".out/", help="default is .out/", type=str)

    args = parser.parse_args()

    main(args)
