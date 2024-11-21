import argparse
import os
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# Suppress specific warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
pd.options.mode.chained_assignment = None  # default='warn'


def normalize_value(value: Any, is_zygosity: bool = False) -> str:
    """Normalize the value for the given task."""
    if isinstance(value, str):
        value = value.lower()
    if value in ["other", "other (see note)"]:
        return "other"
    if is_zygosity and value in ["none", "unknown"]:
        return "unknown"
    return value


def generate_pie_chart(
    merged_df: pd.DataFrame,
    task_name: str,
    title: str,
    conditions: Dict[Callable, Tuple[Callable, str]],
    outdir: str,
    is_zygosity: bool = False,
) -> None:
    """Generate a pie chart for the given task."""
    task_df = merged_df[merged_df["task"].str.lower() == task_name]
    counts, colors = {}, {}

    for _, row in task_df.iterrows():
        truth_value = normalize_value(row["truth_value"], is_zygosity)
        response_combined_1 = normalize_value(row["Response_combined_1"], is_zygosity)
        response_combined_2 = normalize_value(row["Response_combined_2"], is_zygosity)

        for condition, (label_func, color) in conditions.items():
            if condition(truth_value, response_combined_1, response_combined_2):
                label = label_func(truth_value, response_combined_1, response_combined_2)
                if label not in counts:
                    counts[label] = 0
                    colors[label] = color
                counts[label] += 1
                break

    if sum(counts.values()) == 0:
        raise ValueError("No data to plot")

    if len(task_df) != sum(counts.values()):
        raise ValueError(
            f"Not all {task_name} task rows are accounted for in the generated pie chart categories. There are {len(task_df)} rows for this task and only {sum(counts.values())} have been accounted for."
        )

    labels, sizes, colors_list = zip(*[(label, count, colors[label]) for label, count in counts.items() if count > 0])

    plt.rcParams.update({"font.size": 14})

    def autopct_format(values: list) -> Callable[[float], str]:
        def my_format(pct: float) -> str:
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return f"{pct:.1f}%\n({val:d})"

        return my_format

    plt.figure(figsize=(12, 12))
    plt.pie(
        sizes,
        labels=labels,
        colors=colors_list,
        autopct=autopct_format(sizes),
        shadow=True,
        startangle=140,
    )
    plt.title(title)
    plt.axis("equal")
    plt.savefig(os.path.join(outdir, f"{task_name}_piechart.png"), format="png", dpi=300, bbox_inches="tight")


def update_error_analysis_worksheet(df: pd.DataFrame, df_resolved: pd.DataFrame) -> pd.DataFrame:
    """Update the responses in the error analysis DataFrame with the resolved discrepancies."""
    for _, row in df_resolved.iterrows():
        question_number = row["Q###"]
        response = row["Response_1"]
        df.loc[df["Q###"] == question_number, "Response"] = response
    return df


def read_and_process_files(args: argparse.Namespace, output_dir: str) -> pd.DataFrame:
    """Read and process the files to generate the error analysis summary."""
    df_1 = pd.read_csv(args.parsed_ana1_error_analysis_file, sep="\t", encoding="latin1")
    df_2 = pd.read_csv(args.parsed_ana2_error_analysis_file, sep="\t", encoding="latin1")
    df_3 = pd.read_csv(args.parsed_ana3_error_analysis_file, sep="\t", encoding="latin1")
    df_all_qs = pd.read_csv(args.all_sorted_discrep_file, sep="\t", encoding="latin1")

    if args.resolved_discrepancies:
        df_resolved = pd.read_csv(args.resolved_discrep_file, sep="\t", encoding="latin1")

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

    responses_1 = df_1[["Q###", "Response"]].rename(columns={"Q###": "question_number"})
    responses_2 = df_2[["Q###", "Response"]].rename(columns={"Q###": "question_number"})
    responses_3 = df_3[["Q###", "Response"]].rename(columns={"Q###": "question_number"})
    truth_responses = df_all_qs[["question_number", "task", "truth_value"]]

    for df in [responses_1, responses_2, responses_3]:
        df["question_number"] = df["question_number"].str.replace("Q", "").astype(int)

    truth_responses["question_number"] = truth_responses["question_number"].astype(int)

    merged_df = truth_responses.merge(responses_1, on="question_number", how="left")
    merged_df = merged_df.merge(responses_2, on="question_number", how="left", suffixes=("", "_2"))
    merged_df = merged_df.merge(responses_3, on="question_number", how="left", suffixes=("", "_3"))

    merged_df["Response_combined_1"] = merged_df[["Response", "Response_2", "Response_3"]].bfill(axis=1).iloc[:, 0]
    merged_df["Response_combined_2"] = merged_df[["Response", "Response_2", "Response_3"]].bfill(axis=1).iloc[:, 1]

    merged_df = merged_df.drop(columns=["Response", "Response_2", "Response_3"])
    merged_df = merged_df.apply(lambda x: x.str.lower() if x.dtype == "object" else x)

    return merged_df


def main(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.outdir, f"error_analysis_{timestamp}/")
    os.makedirs(output_dir, exist_ok=True)

    merged_df = read_and_process_files(args, output_dir)

    conditions_papers = {
        lambda tv, rc1, rc2: tv == "true"
        and rc1 == "true"
        and rc2 == "true": (lambda tv, rc1, rc2: "EA missed,\n should be in MGT", "#4CAF50"),
        lambda tv, rc1, rc2: tv == "false"
        and rc1 == "false"
        and rc2 == "false": (lambda tv, rc1, rc2: "EA found,\n shouldn't be in MGT", "#2196F3"),
        lambda tv, rc1, rc2: tv == "false"
        and rc1 == "true"
        and rc2 == "true": (lambda tv, rc1, rc2: "EA found,\n should be in MGT", "#FF9800"),
        lambda tv, rc1, rc2: tv == "true"
        and rc1 == "false"
        and rc2 == "false": (lambda tv, rc1, rc2: "EA missed,\n shouldn't be in MGT", "#F44336"),
        lambda tv, rc1, rc2: tv == "true"
        and rc1 == "other"
        and rc2 == "other": (lambda tv, rc1, rc2: "EA missed, turned other", "lightblue"),
        lambda tv, rc1, rc2: tv == "false"
        and rc1 == "other"
        and rc2 == "other": (lambda tv, rc1, rc2: "EA found, turned other", "teal"),
        lambda tv, rc1, rc2: tv == "true"
        and rc1 == "true"
        and rc2 == "false": (lambda tv, rc1, rc2: "EA missed,\n maybe belongs in MGT", "#9C27B0"),
        lambda tv, rc1, rc2: tv == "true"
        and rc1 == "false"
        and rc2 == "true": (lambda tv, rc1, rc2: "EA missed,\n maybe belongs in MGT", "#9C27B0"),
        lambda tv, rc1, rc2: tv == "true"
        and rc1 == "other"
        and rc2 == "true": (lambda tv, rc1, rc2: "EA missed,\n maybe belongs in MGT", "#9C27B0"),
        lambda tv, rc1, rc2: tv == "true"
        and rc1 == "true"
        and rc2 == "other": (lambda tv, rc1, rc2: "EA missed,\n maybe belongs in MGT", "#9C27B0"),
        lambda tv, rc1, rc2: tv == "false"
        and rc1 == "true"
        and rc2 == "false": (lambda tv, rc1, rc2: "EA found,\n maybe doesn't belong in MGT", "pink"),
        lambda tv, rc1, rc2: tv == "false"
        and rc1 == "false"
        and rc2 == "true": (lambda tv, rc1, rc2: "EA found,\n maybe doesn't belong in MGT", "pink"),
        lambda tv, rc1, rc2: tv == "false"
        and rc1 == "other"
        and rc2 == "false": (lambda tv, rc1, rc2: "EA found,\n maybe doesn't belong in MGT", "pink"),
        lambda tv, rc1, rc2: tv == "false"
        and rc1 == "false"
        and rc2 == "other": (lambda tv, rc1, rc2: "EA found,\n maybe doesn't belong in MGT", "pink"),
        lambda tv, rc1, rc2: tv == "true"
        and tv != rc1
        and tv != rc2
        and rc1 != rc2: (lambda tv, rc1, rc2: "EA missed, all dif. responses", "#FFEB3B"),
        lambda tv, rc1, rc2: tv == "false"
        and tv != rc1
        and tv != rc2
        and rc1 != rc2: (lambda tv, rc1, rc2: "EA found, all dif. responses", "#B8860B"),
    }

    conditions_observations = {
        lambda tv, rc1, rc2: tv == "true"
        and rc1 == "this is completely correct"
        and rc2 == "this is completely correct": (lambda tv, rc1, rc2: "EA missed, should be in MGT", "#4CAF50"),
        lambda tv, rc1, rc2: tv == "true"
        and rc1 == "other"
        and rc2 == "other": (lambda tv, rc1, rc2: "EA missed, should be in MGT ->other", "#8BC34A"),
        lambda tv, rc1, rc2: tv == "true"
        and rc1 == "this variant is discussed, but it is not possessed by the specified individual"
        and rc2
        == "this variant is discussed, but it is not possessed by the specified individual": (
            lambda tv, rc1, rc2: "EA missed, shouln't be in MGT. (Var. not connected to indiv.)",
            "#FF9800",
        ),
        lambda tv, rc1, rc2: tv == "true"
        and rc1 == "this variant is either invalid or it is not discussed in the paper"
        and rc2
        == "this variant is either invalid or it is not discussed in the paper": (
            lambda tv, rc1, rc2: "EA missed, shouln't be in MGT. (Var. incorrect)",
            "#FF5722",
        ),
        lambda tv, rc1, rc2: tv == "false"
        and rc1 == "this is completely correct"
        and rc2 == "this is completely correct": (lambda tv, rc1, rc2: "EA found, should be in MGT", "#F44336"),
        lambda tv, rc1, rc2: tv == "false"
        and rc1 == "other"
        and rc2 == "other": (lambda tv, rc1, rc2: "EA found->other", "#9C27B0"),
        lambda tv, rc1, rc2: tv == "false"
        and rc1 == "this variant is discussed, but it is not possessed by the specified individual"
        and rc2
        == "this variant is discussed, but it is not possessed by the specified individual": (
            lambda tv, rc1, rc2: "EA found, shouldn't be in MGT (Var. not connected to indiv.)",
            "#673AB7",
        ),
        lambda tv, rc1, rc2: tv == "false"
        and rc1 == "this variant is either invalid or it is not discussed in the paper"
        and rc2
        == "this variant is either invalid or it is not discussed in the paper": (
            lambda tv, rc1, rc2: "EA found, shouldn't be in MGT (Var. incorrect)",
            "#3F51B5",
        ),
        lambda tv, rc1, rc2: tv == "true": (lambda tv, rc1, rc2: "EA missed, all dif. responses", "#2196F3"),
        lambda tv, rc1, rc2: tv == "false": (lambda tv, rc1, rc2: "EA found, all dif. responses", "#03A9F4"),
    }

    conditions_study_type = {
        lambda tv, rc1, rc2: tv == rc1 and tv == rc2: (lambda tv, rc1, rc2: "EA missed, should be in MGT", "#4CAF50"),
        lambda tv, rc1, rc2: rc1 == rc2 and tv != rc1: (lambda tv, rc1, rc2: f"{tv} -> {rc1}", "#FF5722"),
        lambda tv, rc1, rc2: tv != rc1
        and tv != rc2
        and rc1 != rc2: (lambda tv, rc1, rc2: f"Maybe {tv}, all dif. responses", "#FF9800"),
        lambda tv, rc1, rc2: tv == rc1 or tv == rc2: (lambda tv, rc1, rc2: f"Maybe {tv}", "#03A9F4"),
    }

    conditions_variant_inheritance = conditions_study_type.copy()
    conditions_variant_type = conditions_study_type.copy()

    conditions_zygosity = {
        lambda tv, rc1, rc2: tv == rc1 and tv == rc2: (lambda tv, rc1, rc2: "EA missed,\n should be in MGT", "#4CAF50"),
        lambda tv, rc1, rc2: rc1 == rc2 and tv != rc1: (lambda tv, rc1, rc2: f"{tv} -> {rc1}", "#FF5722"),
        lambda tv, rc1, rc2: tv != rc1 and tv != rc2 and rc1 != rc2: (lambda tv, rc1, rc2: "To Resolve", "#FF9800"),
        lambda tv, rc1, rc2: tv == rc1 or tv == rc2: (lambda tv, rc1, rc2: f"Maybe {tv}", "#03A9F4"),
    }

    generate_pie_chart(merged_df, "papers", "Papers Error Analysis Summary", conditions_papers, output_dir)
    generate_pie_chart(
        merged_df, "observations", "Observation Finding Error Analysis Summary", conditions_observations, output_dir
    )
    generate_pie_chart(merged_df, "study_type", "Study Type Error Analysis Summary", conditions_study_type, output_dir)
    generate_pie_chart(
        merged_df,
        "variant_inheritance",
        "Variant Inheritance Error Analysis Summary",
        conditions_variant_inheritance,
        output_dir,
    )
    generate_pie_chart(
        merged_df, "variant_type", "Variant Type Error Analysis Summary", conditions_variant_type, output_dir
    )
    generate_pie_chart(
        merged_df, "zygosity", "Zygosity Error Analysis Summary", conditions_zygosity, output_dir, is_zygosity=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Error Analysis Summary")
    parser.add_argument(
        "--parsed-ana1-error-analysis-file",
        type=str,
        default="data/error_analysis/parsed_ana1_error_analysis_worksheet.tsv",
        help="Path to parsed analyst 1 error analysis file",
    )
    parser.add_argument(
        "--parsed-ana2-error-analysis-file",
        type=str,
        default="data/error_analysis/parsed_ana2_error_analysis_worksheet.tsv",
        help="Path to parsed analyst 2 error analysis file",
    )
    parser.add_argument(
        "--parsed-ana3-error-analysis-file",
        type=str,
        default="data/error_analysis/parsed_ana3_error_analysis_worksheet.tsv",
        help="Path to parsed analyst 3 error analysis file",
    )
    parser.add_argument(
        "--all-sorted-discrep-file",
        type=str,
        default="data/error_analysis/all_sorted_discrepancies.tsv",
        help="Path to all sorted discrepancies file",
    )
    parser.add_argument(
        "--resolved-discrep-file",
        type=str,
        default="data/error_analysis/resolved_discrepancies/resolved_discrepancies.tsv",
        help="Path to resolved discrepancies file",
    )
    parser.add_argument("--outdir", type=str, default=".out/", help="Output directory")
    parser.add_argument(
        "--resolved-discrepancies",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Include resolved inter-rater discrepancies in the analysis (default: True)",
    )
    args = parser.parse_args()
    main(args)
