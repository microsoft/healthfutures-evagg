"""Script to compare our Evidence Aggregator's paper finding pipeline to the manual ground truth (MGT)  data.

Inputs
- MGT intermediate/train-test split data file (e.g. data/v1/papers_train_v1.tsv)
- .yaml of queries

Outputs:
- benchmarking_paper_finding_results.txt: comparison of the papers that were found to the MGT data
papers
- benchmarking_paper_finding_results.png: barplot of average number of correct, missed, and irrelevant papers
for the Evidence Aggregator tool
- pipeline_mgt_comparison.csv: table comparing the pipeline output to the MGT data
"""

# Libraries
import argparse
import os
import shutil
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from lib.evagg.utils.run import get_previous_run
from scripts.benchmark.utils import get_paper


def calculate_metrics(num_correct: int, num_missed: int, num_irrelevant: int) -> Tuple:
    # Calculate precision and recall from benchmarking results
    # precision is calc as true_positives / (true_positives + false_positives)
    precision = num_correct / (num_correct + num_irrelevant) if num_correct + num_irrelevant != 0 else 0

    # recall is calc as true_positives / (true_positives + false_negatives)
    recall = num_correct / (num_correct + num_missed) if num_correct + num_missed != 0 else 0

    # F1 score is useful when classes are imbalanced
    # If precision and recall are both 0, set F1 score to 0 to avoid ZeroDivisionError
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    return round(precision, 4), round(recall, 4), round(f1_score, 4)


def plot_benchmarking_results(
    output_dir: str,
    joined_df: pd.DataFrame,
) -> None:
    """Plot the benchmarking results from the  set in a barplot."""
    # List of categories
    ea_categories = ["E.A. Correct", "E.A. Missed", "E.A. Irrelevant"]

    num_correct = int((joined_df.in_truth & joined_df.in_pipeline).sum())
    num_missed = int((joined_df.in_truth & ~joined_df.in_pipeline).sum())
    num_irrelevant = int((~joined_df.in_truth & joined_df.in_pipeline).sum())

    values = [num_correct, num_missed, num_irrelevant]

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # Define bar width
    bar_width = 0.35

    # Positions of the left bar boundaries
    bar_l = np.arange(len(values))

    # Positions of the x-axis ticks (center of the bars as bar labels)
    tick_pos = [i + (bar_width / 2) for i in bar_l]

    # Create a bar plot
    bars = ax.bar(bar_l, values, width=bar_width, label="Tool", alpha=0.5, color="b")

    # Set the ticks to be first names
    plt.xticks(tick_pos, ea_categories)
    ax.set_ylabel("Count")
    ax.set_xlabel("Categories")
    plt.legend(loc="upper right")
    plt.title("Evidence Aggregator: #correct, # missed & # irrelevant papers")

    # Function to add value in bar plot labels
    def add_labels(bars: Any) -> None:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_y() + height + 0.9, "%.2f" % height, ha="center", va="bottom"
            )

    # Add appropriate labels
    add_labels(bars)
    plt.xticks(tick_pos, ["Correct", "Missed", "Irrelevant"])

    # Save barplot
    plt.savefig(output_dir + "/benchmarking_paper_finding_results.png")


def build_individual_result_summary(
    tp: pd.DataFrame, fn: pd.DataFrame, fp: pd.DataFrame, row_id: str, prefix: str
) -> str:

    summary = ""

    summary += f"{prefix} # true positives: {tp.shape[0]}\n"
    summary += f"{prefix} # false negatives: {fn.shape[0]}\n"
    summary += f"{prefix} # false positives: {fp.shape[0]}\n\n"

    summary += f"{prefix} true positives:\n"
    for index, row in tp.reset_index().iterrows():
        summary += f"* {index} * {row[row_id]} * {row.paper_title}\n"
    summary += "\n"

    summary += f"{prefix} false negatives:\n"
    for index, row in fn.reset_index().iterrows():
        summary += f"* {index} * {row[row_id]} * {row.paper_title}\n"
    summary += "\n"

    summary += f"{prefix} false positives:\n"
    for index, row in fp.reset_index().iterrows():
        summary += f"* {index} * {row[row_id]} * {row.paper_title}\n"
    summary += "\n"

    return summary


def write_output_summary(
    output_dir: str,
    joined_df: pd.DataFrame,
) -> None:
    ea_tp = joined_df[joined_df.in_truth & joined_df.in_pipeline][["pmid_with_query", "paper_title"]]
    ea_fn = joined_df[joined_df.in_truth & ~joined_df.in_pipeline][["pmid_with_query", "paper_title"]]
    ea_fp = joined_df[~joined_df.in_truth & joined_df.in_pipeline][["pmid_with_query", "paper_title"]]

    ea_precision, ea_recall, ea_f1_score = calculate_metrics(ea_tp.shape[0], ea_fn.shape[0], ea_fp.shape[0])

    # Compile and save the benchmarking results to a file
    with open(os.path.join(output_dir, "benchmarking_paper_finding_results.txt"), "w") as f:
        f.write("Evidence Aggregator Paper Finding Benchmarks Key: \n")
        f.write(
            "- Search 'E.A. overall precision' to see 1) overall Evidence Aggregator precision, recall, F1, and 2) "
            "associated TP, FN, FP paper information. Directly below that, you will see 3) the exact PMIDs and paper "
            "titles in those 3 categories.\n"
        )
        f.write(
            "- Search 'GENE' to gene level stats: 1) TP, FN, FP paper information, and 2) the exact PMIDs and paper "
            "titles in those 3 categories.\n\n"
        )

        f.write(f"E.A. overall precision: {ea_precision} (N_irrelevant = {ea_fp.shape[0]})\n")
        f.write(f"E.A. overall recall: {ea_recall} (N_missed = {ea_fn.shape[0]})\n")
        f.write(f"E.A. overall F1 score: {ea_f1_score}\n\n")

        f.write(build_individual_result_summary(ea_tp, ea_fn, ea_fp, "pmid_with_query", "E.A."))

        # Iterate through all genes and associated PMIDS
        for gene_symbol, gene_df in joined_df.groupby("gene"):

            gene_tp = gene_df[gene_df.in_truth & gene_df.in_pipeline][["pmid", "paper_title"]]
            gene_fn = gene_df[gene_df.in_truth & ~gene_df.in_pipeline][["pmid", "paper_title"]]
            gene_fp = gene_df[~gene_df.in_truth & gene_df.in_pipeline][["pmid", "paper_title"]]

            # Get the gene name from the query
            f.write(f"\nGENE: {gene_symbol}\n")
            f.write(build_individual_result_summary(gene_tp, gene_fn, gene_fp, "pmid", str(gene_symbol)))


def read_queries(yaml_data: Any) -> List[Dict[str, Any]]:
    query_list_yaml = []
    for query in yaml_data:
        # Extract the values, or use an empty string if they're missing
        gene_symbol = query.get("gene_symbol", None)
        min_date = query.get("min_date", "")
        max_date = query.get("max_date", "")
        date_type = query.get("date_type", "")
        retmax = query.get("retmax", "")  # default retmax (i.e. max_papers) is 20

        # Create a new dictionary with the gene_symbol and the values, '' in places where .yaml does not show a value
        new_dict = {
            "gene_symbol": gene_symbol,
            "min_date": min_date,
            "max_date": max_date,
            "date_type": date_type,
            "retmax": retmax,
        }

        # Add the new dictionary to the list
        query_list_yaml.append(new_dict)
    return query_list_yaml


def get_paper_title(pmid: str) -> str:
    # This increases execution time a fair amount, we could alternatively store the titles in the MGT and the pipeline
    # output to speed things up if we wanted.
    try:
        paper = get_paper(pmid)
        return paper.props.get("title", "Unknown") if paper else "Unknown"
    except Exception as e:
        print(f"Error getting title for paper {pmid}: {e}")

    return "Unknown"


def get_mgt(args: argparse.Namespace) -> pd.DataFrame:
    # Read the intermediate manual ground truth (MGT) data file from the TSV file
    mgt_df = pd.read_csv(args.mgt_train_test_path, sep="\t")
    print("Number of manual ground truth pmids: ", mgt_df.shape[0] - 1)

    # Filter to only papers where the "can_access" column is True
    if args.mgt_full_text_only:
        mgt_df = mgt_df.query("can_access == True")
        print("Only considering full text papers pmids: ", mgt_df.shape[0] - 1)

    # Filter to exclude skipped_pmids
    if args.skipped_pmids:
        with open(args.skipped_pmids, "r") as file:
            skipped_pmids = [line.strip() for line in file.readlines()]
        mgt_df = mgt_df[~mgt_df.pmid.astype(str).isin(skipped_pmids)]
        print("Not considering skipped pmids: ", mgt_df.shape[0] - 1)

    # The MGT df isn't perfect and sometimes contains duplicates
    mgt_df.drop_duplicates(subset=["gene", "pmid"], inplace=True)

    return mgt_df


def get_pipeline_output(args: argparse.Namespace) -> Tuple[str, pd.DataFrame]:

    if args.pipeline_output:
        run_record = get_previous_run("evagg_pipeline", name_includes=args.pipeline_output)
    else:
        run_record = get_previous_run("evagg_pipeline")

    if not run_record or not run_record.path or not run_record.output_file:
        raise ValueError("No pipeline output found.")

    # Read in the corresponding pipeline output. Assume we're running from repo root.
    pipeline_df = pd.read_csv(os.path.join(run_record.path, run_record.output_file), sep="\t", skiprows=1)

    # If paper_id is prefixed with "pmid:", remove it.
    pipeline_df["pmid"] = pipeline_df["paper_id"].str.lstrip("pmid:").astype(int)

    # Only keep the columns we care about.
    pipeline_df = pipeline_df[["gene", "pmid"]]

    # We only need one of each paper/gene pair, so we drop duplicates.
    pipeline_df = pipeline_df.drop_duplicates(subset=["gene", "pmid"])

    if not args.skip_validation:
        # Get the query/ies from config file so we know the list of genes that were processed.
        # If there is an override for queries on the command line, we'll need to respect that.
        override_strings = [
            run_record.args[i + 1] for i, arg in enumerate(run_record.args) if arg == "--override" or arg == "-o"
        ]

        if _ := next((s for s in override_strings if s.startswith("queries:")), None):
            raise NotImplementedError("queries: override not yet implemented.")
        elif query_yaml := next((s for s in override_strings if s.startswith("queries.di_factory:")), None):
            with open(query_yaml.split(":")[1], "r") as file:
                yaml_data = yaml.safe_load(file)
                query_list_yaml = read_queries(yaml_data)
        else:
            # No override for queries, so we'll use what we find in the pipeline output
            with open(run_record.args[0], "r") as file:
                yaml_data = yaml.safe_load(file)

            if ".yaml" in str(yaml_data["queries"]):  # leading to query .yaml
                with open(yaml_data["queries"]["di_factory"], "r") as file:
                    yaml_data = yaml.safe_load(file)
                    query_list_yaml = read_queries(yaml_data)
            elif yaml_data["queries"] is not None:
                query_list_yaml = read_queries(yaml_data["queries"])
            else:
                raise ValueError("No queries found in the .yaml file.")

        yaml_genes = [query["gene_symbol"] for query in query_list_yaml]

        if any(x not in yaml_genes for x in pipeline_df.gene.unique().tolist()):
            raise ValueError("Gene(s) in pipeline output .tsv not found in the .yaml file.")

    return (run_record.path, pipeline_df)


def main(args: argparse.Namespace) -> None:

    # Gather inputs.
    mgt_df = get_mgt(args)
    pipeline_path, pipeline_df = get_pipeline_output(args)

    # Join mgt_df and pipeline_df on (gene, pmid), keeping all rows and keeping track of whether a row was present in
    # either of the source dataframes.
    joined_df = pd.merge(
        mgt_df,
        pipeline_df,
        how="outer",
        left_on=["gene", "pmid"],
        right_on=["gene", "pmid"],
        indicator=True,
    )

    joined_df["in_truth"] = joined_df._merge.isin(["both", "left_only"])
    joined_df["in_pipeline"] = joined_df._merge.isin(["both", "right_only"])
    joined_df.drop(columns=["_merge"], inplace=True)
    joined_df["pmid_with_query"] = joined_df["pmid"].astype(str) + " (" + joined_df["gene"] + ")"

    joined_df["paper_title"] = joined_df["pmid"].apply(get_paper_title)

    outdir = args.outdir or pipeline_path + "_paper_finding_benchmarks"

    # Prep output directory.
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)

    # Plot results.
    plot_benchmarking_results(outdir, joined_df)

    # Write text output summary.
    write_output_summary(outdir, joined_df)

    # Write the joined_df to file.
    joined_df.to_csv(os.path.join(outdir, "pipeline_mgt_comparison.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evidence Aggregator Paper Finding Benchmarks")
    parser.add_argument(
        "-p",
        "--pipeline-output",
        nargs="?",
        default="",
        type=str,
        help="Path to the output directory corresponding to the pipeline run. "
        + "If none is provided, the most recent output for 'run_evagg_pipeline' will be used.",
    )
    parser.add_argument(
        "-m",
        "--mgt-train-test-path",
        nargs="?",
        default="data/v1/papers_train_v1.tsv",
        type=str,
        help="Default is data/v1/papers_train_v1.tsv.",
    )
    parser.add_argument(
        "-s",
        "--skipped-pmids",
        nargs="?",
        default="",
        type=str,
        help="Path to file containing pmids that should be removed from the truth set before running benchmarks. "
        + "If none is provided then all pmids will be considered.",
    )
    parser.add_argument(
        "-f",
        "--mgt-full-text-only",
        nargs="?",
        default=True,
        type=bool,
        help="Flag to subset mgt papers under consideration to only those will full text available. Default is True.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default="",
        type=str,
        help=(
            "Results output directory. Default defaults to the pipeline input directory with _paper_finding appended."
        ),
    )
    parser.add_argument(
        "-v",
        "--skip-validation",
        action="store_true",
        help="Flag to skip validation that the genes in the pipeline output are present in the .yaml file. "
        + "Default is False.",
    )
    args = parser.parse_args()

    print("Evidence Aggregator Paper Finding Benchmarks:")
    for arg, value in vars(args).items():
        print(f"- {arg}: {value}")

    print("\n")

    main(args)
