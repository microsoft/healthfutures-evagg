# %% Imports.

import os
from datetime import datetime
from functools import cache
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from scripts.truthset.error_analysis.utils import parse_error_analysis_excel

# %% Constants.

ANALYST_1_FILE = "data/error_analysis/ana1_error_analysis_worksheet.csv"
ANALYST_2_FILE = "data/error_analysis/ana2_error_analysis_worksheet.csv"
ANALYST_3_FILE = "data/error_analysis/ana3_error_analysis_worksheet.csv"
ALL_SORTED_DISCREP_FILE = "data/error_analysis/all_sorted_discrepancies.tsv"
RESOLVED_DISCREP_FILE = "data/error_analysis/resolved_discrepancies.tsv"
SKIPPED_PMIDS_FILE = "scripts/benchmark/paper_finding/paper_finding_benchmarks_skipped_pmids.txt"
OUTDIR = ".out/"
RESOLVED_DISCREPANCIES = True

# %% Functions.


def update_error_analysis_worksheet(df: pd.DataFrame, df_resolved: pd.DataFrame) -> pd.DataFrame:
    """Update the responses in the error analysis DataFrame with the resolved discrepancies."""
    for _, row in df_resolved.iterrows():
        question_number = row["Q###"]
        phenotype = row["Phenotype_1"]
        response = row["Response_1"]
        if phenotype:
            match = (df["Q###"] == question_number) & (df["Phenotype"] == phenotype)
        else:
            match = df["Q###"] == question_number
        df.loc[match, "Response"] = response
    return df


@cache
def load_skipped_pmids(skipped_pmids_file: str) -> list[int]:
    """Load the skipped PMIDs from the file."""
    print("Loading skipped PMIDs...")
    with open(skipped_pmids_file) as f:
        return [int(line.strip()) for line in f.readlines()]


def remove_skipped_pmids_by_question(df: pd.DataFrame, question_col: str = "Questions") -> pd.DataFrame:
    """Remove the skipped PMIDs from the DataFrame."""
    skipped_pmids = load_skipped_pmids(SKIPPED_PMIDS_FILE)

    # For any value in question_col that looks like "The paper (\d+)...", extract the number and convert to int
    # otherwise fill with 0
    pmids = (
        df[question_col]
        .str.extract(r"The paper (\d+) discusses one or more human genetic variants", expand=False)
        .fillna(0)
        .astype(int)
    )
    return df[~pmids.isin(skipped_pmids)]


def remove_skipped_pmids_by_pmid(df: pd.DataFrame, pmid_col: str = "pmid") -> pd.DataFrame:
    """Remove the skipped PMIDs from the DataFrame."""
    skipped_pmids = load_skipped_pmids(SKIPPED_PMIDS_FILE)
    return df[~df[pmid_col].isin(skipped_pmids)]


def read_and_process_files(output_dir: str) -> pd.DataFrame:
    """Read and process the files to generate the error analysis summary."""
    df_1 = parse_error_analysis_excel(ANALYST_1_FILE)
    df_1 = remove_skipped_pmids_by_question(df_1)
    df_2 = parse_error_analysis_excel(ANALYST_2_FILE)
    df_2 = remove_skipped_pmids_by_question(df_2)
    df_3 = parse_error_analysis_excel(ANALYST_3_FILE)
    df_3 = remove_skipped_pmids_by_question(df_3)

    df_all_qs = pd.read_csv(ALL_SORTED_DISCREP_FILE, sep="\t", encoding="latin1")
    df_all_qs = remove_skipped_pmids_by_pmid(df_all_qs)

    # This is a little messy, but because all_sorted_discrep_file only contains one row for each observation, and
    # we asked reviewers to look at each phenotype term separately, we need to create a separate row for each phenotype
    # question we would have asked.
    def flatten_unique(input: List[List[str]]) -> List[str]:
        return list({item for sublist in input for item in sublist})

    # First, get the phenotype terms that were either found only in truth or in output as lists of lists.
    df_all_qs["truth_dict_values"] = df_all_qs["truth_dict"].apply(
        lambda x: list(eval(x).values()) if isinstance(x, str) else []
    )
    df_all_qs["output_dict_values"] = df_all_qs["output_dict"].apply(
        lambda x: list(eval(x).values()) if isinstance(x, str) else []
    )
    # Combine those lists of lists.
    df_all_qs["phenotype"] = df_all_qs["truth_dict_values"] + df_all_qs["output_dict_values"]
    # Flatten and remove duplicates.
    df_all_qs["phenotype"] = df_all_qs["phenotype"].apply(flatten_unique)
    # Make new rows for each unique phenotype term.
    df_all_qs = df_all_qs.explode("phenotype").reset_index(drop=True)
    # Finally, assign a truth_value of True if the term is in truth_dict_values and False if not.
    # Only change the phenotype questions.
    df_all_qs["truth_value"] = df_all_qs.apply(
        lambda x: (
            (True if x["phenotype"] in flatten_unique(x["truth_dict_values"]) else False)
            if x["task"] == "phenotype"
            else x["truth_value"]
        ),
        axis=1,
    )

    df_all_qs["phenotype"] = df_all_qs["phenotype"].fillna("")
    truth_responses = df_all_qs[["question_number", "phenotype", "task", "truth_value"]].copy()

    if RESOLVED_DISCREPANCIES:
        df_resolved = pd.read_csv(RESOLVED_DISCREP_FILE, sep="\t", encoding="latin1")
        df_resolved = remove_skipped_pmids_by_question(df_resolved, question_col="Questions_1")
        df_resolved["Phenotype_1"] = df_resolved["Phenotype_1"].fillna("")

        # Ensure that all inter-rater discrepancies are resolved
        assert (df_resolved["Response_1"] == df_resolved["Response_2"]).all()

        # Update the error analysis worksheets with the resolved inter-rater discrepancies
        df_1 = update_error_analysis_worksheet(df_1, df_resolved)
        df_2 = update_error_analysis_worksheet(df_2, df_resolved)
        df_3 = update_error_analysis_worksheet(df_3, df_resolved)

        # Save those updated worksheets
        df_1.to_csv(output_dir + "parsed_ana1_error_analysis_worksheet_resolved.tsv", sep="\t", index=False)
        df_2.to_csv(output_dir + "parsed_ana2_error_analysis_worksheet_resolved.tsv", sep="\t", index=False)
        df_3.to_csv(output_dir + "parsed_ana3_error_analysis_worksheet_resolved.tsv", sep="\t", index=False)

    responses_1 = df_1[["Q###", "Response", "Phenotype"]].rename(
        columns={"Q###": "question_number", "Phenotype": "phenotype"}
    )
    responses_2 = df_2[["Q###", "Response", "Phenotype"]].rename(
        columns={"Q###": "question_number", "Phenotype": "phenotype"}
    )
    responses_3 = df_3[["Q###", "Response", "Phenotype"]].rename(
        columns={"Q###": "question_number", "Phenotype": "phenotype"}
    )

    for df in [responses_1, responses_2, responses_3]:
        df["question_number"] = df["question_number"].str.replace("Q", "").astype(int)
        df["phenotype"] = df["phenotype"].apply(lambda x: x.split("(")[1].rstrip(")") if "(" in x else x)

    truth_responses["question_number"] = truth_responses["question_number"].astype(int)

    merged_df = truth_responses.merge(responses_1, on=["question_number", "phenotype"], how="left")
    merged_df = merged_df.merge(responses_2, on=["question_number", "phenotype"], how="left", suffixes=("", "_2"))
    merged_df = merged_df.merge(responses_3, on=["question_number", "phenotype"], how="left", suffixes=("", "_3"))

    # At this point, every row should have exactly two non-nan values in the Response columns and those values should
    # be the same. Verify this.
    assert (merged_df[["Response", "Response_2", "Response_3"]].notna().sum(axis=1) == 2).all()
    assert (merged_df[["Response", "Response_2", "Response_3"]].nunique(axis=1) == 1).all()

    # Collect the single response value into a new column and drop the individual response columns
    # To do this we can use "Response" if it's not nan and "Response_2" otherwise, because of the checks above
    merged_df["response"] = merged_df["Response"].fillna(merged_df["Response_2"])

    # Let's recode some values.
    # Any case where response is "TRUE" or "FALSE", convert it to the appropriate boolean value
    merged_df["response"] = merged_df["response"].str.lower().replace({"true": True, "false": False})
    merged_df["response"] = merged_df["response"].replace(
        {
            "this is completely correct": True,
            "this variant is discussed, but it is not possessed by the specified individual": False,
            "this variant is either invalid or it is not discussed in the paper": False,
        }
    )

    # Any case where truth_value is "TRUE" or "FALSE", convert it to the appropriate boolean value. Only do this when
    # truth_value is a string.
    is_str = merged_df["truth_value"].apply(lambda x: isinstance(x, str))
    merged_df.loc[is_str, "truth_value"] = (
        merged_df.loc[is_str, "truth_value"].str.lower().replace({"true": True, "false": False})
    )

    # Drop Response, Response_2, and Response_3 columns
    merged_df.drop(columns=["Response", "Response_2", "Response_3"], inplace=True)

    return merged_df


# %% Load in data.

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(OUTDIR, f"error_analysis_{timestamp}/")
os.makedirs(output_dir, exist_ok=True)

merged_df = read_and_process_files(output_dir)
# Drop the study_type task
merged_df = merged_df.query("task != 'study_type'")

# %% Make barplots

for task in ["papers", "observations"]:
    task_df = merged_df.query("task == @task").copy()

    # Make a stacked bar plot based on the truth value and response value
    # First, we need to normalize the values

    # Make a new column for error_type, where error_type is "FP" if truth_value is False and error_type is "FN"
    # if truth_value is True
    if task == "papers":
        task_df["Error type"] = task_df["truth_value"].apply(
            lambda x: "Putative\nmissed papers" if x else "Putative\nirrelevant papers"
        )
    else:
        task_df["Error type"] = task_df["truth_value"].apply(
            lambda x: "Putative\nmissed obs." if x else "Putative\nirrelevant obs."
        )

    # Make a new column truth modification, where truth modification is True if truth_value is False and response is
    # True or vice versa
    task_df["Truth update"] = task_df.apply(lambda x: x["truth_value"] != x["response"], axis=1)

    import seaborn as sns

    # Now we can make the plot, make sure to add counts to the plot
    sns.set_theme(style="whitegrid")
    # make the stacked bar plot, use the following palette:
    #     palette={"False": "#1F77B4", "True": "#FA621E"},
    g = (
        task_df.groupby(["Error type", "Truth update"])
        .size()
        .unstack()
        .plot(kind="bar", stacked=True, color=["#1F77B4", "#FA621E"])
    )
    # set the fig size
    plt.gcf().set_size_inches(4, 4)

    # set xlabel orientation to horizontal
    plt.xticks(rotation=0)

    plt.ylabel("Count")

    if task == "papers":
        plt.title("Paper selection error analysis")
    else:
        plt.title("Observation finding error analysis")

    plt.savefig(os.path.join(output_dir, f"{task}_barplot.png"), format="png", dpi=300, bbox_inches="tight")

# %% Also make a single barplot for the different content tasks.

content_tasks = ["zygosity", "variant_inheritance", "variant_type", "phenotype"]

# Again this should be stacked bar plot with one bar for each of the content_tasks values, which correspond to values
# of task in merged_df. The bars should be colored by truth_update, which is true if truth_value and response differ.
# The y-axis should be the count of each combination of content_task and truth_update.

content_df = merged_df.query("task in @content_tasks").copy()

content_df["Truth update"] = content_df.apply(lambda x: x["truth_value"] != x["response"], axis=1)

plt.figure(figsize=(12, 8))
g = (
    content_df.groupby(["task", "Truth update"])
    .size()
    .unstack()
    .plot(kind="bar", stacked=True, color=["#1F77B4", "#FA621E"])
)

plt.title("Content extraction error analysis")
plt.ylabel("Count")
plt.xlabel("Task")
# In the xticks, replace underscores with escaped newlines and capitalize.
plt.xticks([0, 1, 2, 3], ["Phenotype", "Variant\ninheritance", "Variant\ntype", "Zygosity"])

# rotate the xticks
plt.xticks(rotation=0)


plt.savefig(os.path.join(output_dir, "content_extraction_barplot.png"), format="png", dpi=300, bbox_inches="tight")

# %% Write out some numbers.

merged_df["update"] = merged_df["truth_value"] != merged_df["response"]

n_total = merged_df.shape[0]
print(f"Total number of disagreements: {n_total}")

for task in ["papers", "observations", "zygosity", "variant_inheritance", "variant_type", "phenotype"]:
    n_task = merged_df.query("task == @task").shape[0]
    print(f"For the task {task}:")
    print(f"  Total number of disagreements: {n_task} ({n_task / n_total * 100:.2f}%)")
    print(
        f"  Number of responses that were updated: {merged_df.query('task == @task and update').shape[0]} "
        f"({merged_df.query('task == @task and update').shape[0] / n_task * 100:.2f}%)"
    )

print("For all tasks:")
print(f"  Total number of responses: {merged_df.shape[0]}")
print(
    f"  Number of responses that were updated: "
    f"{merged_df.query('update').shape[0]} ({merged_df.query('update').shape[0] / merged_df.shape[0] * 100:.2f}%)"
)


# %% Digging in

pheno_df = merged_df.query("task == 'phenotype'")

pheno_df["update"].mean()

# %%
