"""Script to compare our Evidence Aggregator's paper finding pipeline to the manual ground truth (MGT)  data.

Inputs
- MGT intermediate/train-test split data file (e.g. data/v1/papers_train_v1.tsv)
- .yaml of queries

Outputs:
- benchmarking_paper_finding_results_train.txt: comparison of the papers that were found to the MGT data
papers
- benchmarking_paper_finding_results_train.png: barplot of average number of correct, missed, and irrelevant papers
for the Evidence Aggregator tool and PubMed
- filtering_paper_finding_results_train.png: barplot of average number of papers filtered into rare and
other categories
"""

# Libraries
import argparse
import glob
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime
from functools import cache
from typing import Dict, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from lib.di import DiContainer
from lib.evagg.ref import IPaperLookupClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_git_commit_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def read_queries(yaml_data):
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


def get_ground_truth_pmids(gene, mgt_gene_pmids_dict):
    """Get the manual ground truth (MGT)  data PMIDs for a gene.

    Return the PMIDs for the gene if it is in the ground truth  data, otherwise return None.
    """
    if gene in mgt_gene_pmids_dict.keys():
        return mgt_gene_pmids_dict[gene]
    else:
        return None


def compare_pmid_lists(input_pmids, truth_pmids):
    """Compare the PMIDs that were found either by ev. agg. or competing tool (e.g. PubMed) to the MGT PMIDs.

    Args:
        input_pmids (list): the ids of papers that were found
        truth_pmids (list): the ids of papers from the MGT data

    Returns:
        correct_pmids (list): the PMIDs of the papers that were found that match the MGT data papers
        missed_pmids (list): the PMIDs of the papers that are in the MGT data papers but not in the
        papers that were found
        irrelevant_pmids (list): the PMIDs of the papers that are in the papers that were found but not in
        the MGT data papers
    """
    correct_pmids = list(set(input_pmids).intersection(set(truth_pmids)))
    missed_pmids = list(set(truth_pmids).difference(set(input_pmids)))
    irrelevant_pmids = list(set(input_pmids).difference(set(truth_pmids)))

    return correct_pmids, missed_pmids, irrelevant_pmids


def plot_benchmarking_results_tool_only(
    benchmarking_train_results,
):  # TODO: consider merging this function with plot_benchmarking_results, given the code similarity
    """Plot the benchmarking results from the  set in a barplot.

    Return the results dataframe (see keys below) for subsequent plotting.
    """
    keys = [
        "Gene",
        "Rare Disease Papers",
        "Other Papers",
        "Conflicting Papers",
        "E.A. Correct",
        "E.A. Missed",
        "E.A. Irrelevant",
    ]
    results_to_plot = {key: [] for key in keys}

    for gene, values in benchmarking_train_results.items():
        for i, key in enumerate(keys):
            if key == "Gene":
                results_to_plot[key].append(gene)
            else:
                results_to_plot[key].append(values[i - 1])

    results_to_plot = pd.DataFrame(results_to_plot)

    # List of categories
    ea_categories = ["E.A. Correct", "E.A. Missed", "E.A. Irrelevant"]

    # Calculate averages
    tool_averages = results_to_plot[ea_categories].mean()

    # Calculate standard deviations
    tool_std = results_to_plot[ea_categories].std()

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # Define bar width
    bar_width = 0.35

    # Positions of the left bar boundaries
    bar_l = np.arange(len(tool_averages))

    # Positions of the x-axis ticks (center of the bars as bar labels)
    tick_pos = [i + (bar_width / 2) for i in bar_l]

    # Create a bar plot for 'Tool'
    tool_bars = ax.bar(
        bar_l,
        tool_averages,
        width=bar_width,
        label="Tool",
        alpha=0.5,
        color="b",
        yerr=tool_std,
    )

    # Set the ticks to be first names
    plt.xticks(tick_pos, list(tool_averages.index))
    ax.set_ylabel("Average")
    ax.set_xlabel("Categories")
    plt.legend(loc="upper right")
    plt.title("Avg. # correct, missed & irrelevant papers: E.A.")

    # Function to add average value in bar plot labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_y() + height + 0.9, "%.2f" % height, ha="left", va="bottom"
            )

    # Add appropriate labels
    add_labels(tool_bars)
    plt.xticks(tick_pos, ["Correct", "Missed", "Irrelevant"])

    # Save barplot
    plt.savefig(args.outdir + "/benchmarking_paper_finding_results_train.png")
    return results_to_plot


def plot_benchmarking_results(benchmarking_train_results):
    """Plot the benchmarking results from the  set in a barplot.

    Return the results dataframe (see keys below) for subsequent plotting.
    """
    keys = [
        "Gene",
        "Rare Disease Papers",
        "Other Papers",
        "Conflicting Papers",
        "E.A. Correct",
        "E.A. Missed",
        "E.A. Irrelevant",
        "PubMed Correct",
        "PubMed Missed",
        "PubMed Irrelevant",
    ]
    results_to_plot = {key: [] for key in keys}

    for gene, values in benchmarking_train_results.items():
        for i, key in enumerate(keys):
            if key == "Gene":
                results_to_plot[key].append(gene)
            else:
                results_to_plot[key].append(values[i - 1])

    results_to_plot = pd.DataFrame(results_to_plot)

    # List of categories
    ea_categories = ["E.A. Correct", "E.A. Missed", "E.A. Irrelevant"]
    pubmed_categories = ["PubMed Correct", "PubMed Missed", "PubMed Irrelevant"]

    # Calculate averages
    tool_averages = results_to_plot[ea_categories].mean()
    pubmed_averages = results_to_plot[pubmed_categories].mean()

    # Calculate standard deviations
    tool_std = results_to_plot[ea_categories].std()
    pubmed_std = results_to_plot[pubmed_categories].std()

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # Define bar width
    bar_width = 0.35

    # Positions of the left bar boundaries
    bar_l = np.arange(len(tool_averages))

    # Positions of the x-axis ticks (center of the bars as bar labels)
    tick_pos = [i + (bar_width / 2) for i in bar_l]

    # Create a bar plot for 'Tool'
    tool_bars = ax.bar(
        bar_l,
        tool_averages,
        width=bar_width,
        label="Tool",
        alpha=0.5,
        color="b",
        yerr=tool_std,
    )

    # Create a bar plot for 'PubMed'
    pubmed_bars = ax.bar(
        bar_l + bar_width,
        pubmed_averages,
        width=bar_width,
        label="PubMed",
        alpha=0.5,
        color="r",
        yerr=pubmed_std,
    )

    # Set the ticks to be first names
    plt.xticks(tick_pos, list(tool_averages.index))
    ax.set_ylabel("Average")
    ax.set_xlabel("Categories")
    plt.legend(loc="upper right")
    plt.title("Avg. # correct, missed & irrelevant papers: E.A. vs. PubMed")

    # Function to add average value in bar plot labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_y() + height + 0.9, "%.2f" % height, ha="left", va="bottom"
            )

    # Add appropriate labels
    add_labels(tool_bars)
    add_labels(pubmed_bars)
    plt.xticks(tick_pos, ["Correct", "Missed", "Irrelevant"])

    # Save barplot
    plt.savefig(args.outdir + "/benchmarking_paper_finding_results_train.png")
    return results_to_plot


# Plot the filtereds into paper 4 categories
def plot_filtered_categories(results_to_plot):
    """Plot the filtered results (2 categories: rare disease, other) in a barplot."""
    # Calculate averages
    rare_disease_avg = results_to_plot["Rare Disease Papers"].mean()
    other_papers_avg = results_to_plot["Other Papers"].mean()
    conflicting = results_to_plot["Conflicting Papers"].mean()

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # Define bar width
    bar_width = 0.35

    # Positions of the left bar boundaries
    bar_l = np.arange(3)

    # Positions of the x-axis ticks (center of the bars as bar labels)
    tick_pos = [i + (bar_width / 2) for i in bar_l]

    # Create a bar plot
    bars = ax.bar(
        bar_l,
        [rare_disease_avg, other_papers_avg, conflicting],
        width=bar_width,
        alpha=0.5,
        color="b",
    )

    # Set the ticks to be first names
    plt.xticks(tick_pos, ["Rare Disease Papers", "Other Papers", "Conflicting Papers"], rotation=10)
    ax.set_ylabel("Average")
    ax.set_xlabel("Categories")
    plt.title("Avg. # PubMed papers filtered into rare and other categories")

    # Function to add labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + height + 0.5,
                "%.2f" % float(height),
                ha="center",
                va="bottom",
            )

    # Call the function for each barplot
    add_labels(bars)

    plt.savefig(args.outdir + "/filtering_paper_finding_results_train.png")


def calculate_metrics(num_correct, num_missed, num_irrelevant) -> tuple:
    # Calculate precision and recall from benchmarking results
    # precision is calc as true_positives / (true_positives + false_positives)
    precision = len(num_correct) / (len(num_correct) + len(num_irrelevant))

    # recall is calc as true_positives / (true_positives + false_negatives)
    recall = len(num_correct) / (len(num_correct) + len(num_missed))

    # F1 score is useful when classes are imbalanced
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score


@cache
def get_lookup_client() -> IPaperLookupClient:
    ncbi_lookup: IPaperLookupClient = DiContainer().create_instance({"di_factory": "lib/config/ncbi_lookup.yaml"}, {})
    return ncbi_lookup


def get_paper_titles(pmids: Set[str]) -> Dict[str, str]:
    # This increases execution time a fair amount, we could alternatively store the titles in the MGT and the pipeline
    # output to speed things up if we wanted.
    client = get_lookup_client()
    titles = {}
    for pmid in pmids:
        try:
            paper = client.fetch(pmid)
            titles[pmid] = paper.props.get("title", "Unknown") if paper else "Unknown"
        except Exception as e:
            print(f"Error getting title for paper {pmid}: {e}")
            titles[pmid] = "Unknown"
    return titles


def main(args):

    # Open and load the .yaml file
    with open(args.library_config, "r") as file:
        yaml_data = yaml.safe_load(file)

    # Read the intermediate manual ground truth (MGT) data file from the TSV file
    mgt_df = pd.read_csv(args.mgt_train_test_path, sep="\t")
    print("Number of manual ground truth pmids: ", mgt_df.shape[0] - 1)

    # Filter to only papers where the "has_fulltext" column is True
    if args.mgt_full_text_only:
        mgt_df = mgt_df[mgt_df["has_fulltext"] == True]
        print("Only considering full text papers pmids: ", mgt_df.shape[0] - 1)

    # Get the query/ies from .yaml file so we know the list of genes processed.
    if ".yaml" in str(yaml_data["queries"]):  # leading to query .yaml
        with open(yaml_data["queries"]["di_factory"], "r") as file:
            yaml_data = yaml.safe_load(file)
            query_list_yaml = read_queries(yaml_data)
    elif yaml_data["queries"] is not None:
        query_list_yaml = read_queries(yaml_data["queries"])
    else:
        raise ValueError("No queries found in the .yaml file.")

    yaml_genes = [query["gene_symbol"] for query in query_list_yaml]

    # Read in the corresponding pipeline output.
    pipeline_df = pd.read_csv(args.pipeline_output, sep="\t", skiprows=1)
    if "paper_disease_category" in pipeline_df.columns:
        # Create rare disease pipeline df
        pipeline_df = pipeline_df[pipeline_df["paper_disease_category"] == "rare disease"]
    if "paper_disease_category" not in pipeline_df.columns:
        pipeline_df["paper_disease_category"] = "rare disease"
    if "paper_disease_categorizations" not in pipeline_df.columns:
        pipeline_df["paper_disease_categorizations"] = "{}"

    # We only need one of each paper/gene pair, so we drop duplicates.
    pipeline_df = pipeline_df.drop_duplicates(subset=["gene", "paper_id"])

    if any(x not in yaml_genes for x in pipeline_df.gene.unique().tolist()):
        raise ValueError("Gene(s) in pipeline output not found in the .yaml file.")

    # Initialize benchmarking results dictionary
    benchmarking_results = {}

    # For each query, get papers, compare ev. agg. papers to MGT data papers,
    # compare PubMed papers to MGT data papers. Write results to benchmarking against MGT file.
    if os.path.isdir(args.outdir):
        shutil.rmtree(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    # Save library output table (Evidence Aggregator table) to the same output directory
    shutil.copy(args.pipeline_output, args.outdir)

    # Move the paper finding directions, process, and/or few shot prompts into the benchmarking directory
    directions_files = glob.glob("lib/evagg/content/prompts/paper_finding_directions_*.txt")
    for file in directions_files:
        shutil.move(file, args.outdir)
    process_files = glob.glob("lib/evagg/content/prompts/paper_finding_process_*.txt")
    for file in process_files:
        shutil.move(file, args.outdir)
    few_shot_files = glob.glob("lib/evagg/content/prompts/paper_finding_few_shot_*.txt")
    for file in few_shot_files:
        shutil.move(file, args.outdir)
    full_text__files = glob.glob("lib/evagg/content/prompts/paper_finding_full_text_directions_*.txt")
    for file in full_text__files:
        shutil.move(file, args.outdir)

    # Compute overall precision and recall prior to gene-specific analysis
    truth_pmids = set(mgt_df[mgt_df["has_fulltext"] == True].pmid)
    pipeline_pmids = set(pipeline_df["paper_id"].str.lstrip("pmid:").astype(int))

    # Calculate the false positives
    overall_false_positives = pipeline_pmids.difference(truth_pmids)

    overall_false_negatives = truth_pmids.difference(pipeline_pmids)

    overall_true_positives = pipeline_pmids.intersection(truth_pmids)

    precision = len(overall_true_positives) / (len(overall_true_positives) + len(overall_false_positives))
    recall = len(overall_true_positives) / (len(overall_true_positives) + len(overall_false_negatives))

    # Assert that the tp + fp + fn = pipeline_pmids
    assert len(overall_true_positives) + len(overall_false_positives) == len(pipeline_pmids)

    # Assert that tp + fn = truth_pmids
    assert len(overall_true_positives) + len(overall_false_negatives) == len(truth_pmids)

    # Compile and save the benchmarking results to a file
    with open(os.path.join(args.outdir, "benchmarking_paper_finding_results_train.txt"), "w") as f:
        f.write(f"\nOverall Precision: {precision} (N_irrelevant = {len(overall_false_positives)})")
        f.write(f"\nOverall Recall: {recall} (N_missed = {len(overall_false_negatives)})")
        f.write(f"\n true positives: {len(overall_true_positives)}")
        f.write(f"\n false negatives: {len(overall_false_negatives)}")
        f.write(f"\n false positives: {len(overall_false_positives)}\n")

        # Write out the true positives
        f.write(f"\nTrue positives:\n")
        for i, p in enumerate(overall_true_positives):
            f.write(f"* {i + 1} * {p} * {get_paper_titles({p})[p]}\n")

        # Write out the false negatives
        f.write(f"\nFalse negatives:\n")
        for i, p in enumerate(overall_false_negatives):
            f.write(f"* {i + 1} * {p} * {get_paper_titles({p})[p]}\n")

        # Write out the false positives
        f.write(f"\nFalse positives:\n")
        for i, p in enumerate(overall_false_positives):
            f.write(f"* {i + 1} * {p} * {get_paper_titles({p})[p]}\n")

        # Iterate through all genes and associated PMIDS
        for term, gene_df in pipeline_df.groupby("gene"):

            # Get the gene name from the query
            f.write(f"\nGENE: {term}\n")
            print("Analyzing found papers for: ", term, "...")

            # Initialize the list for this gene
            if args.isolated_run:
                benchmarking_results[term] = [0] * 9
            else:
                benchmarking_results[term] = [0] * 6

            # Get pmids from pipeline output for this gene.
            rare_disease_ids = (
                gene_df[gene_df["paper_disease_category"] == "rare disease"]["paper_id"].str.lstrip("pmid:").tolist()
            )
            other_ids = gene_df[gene_df["paper_disease_category"] == "other"]["paper_id"].str.lstrip("pmid:").tolist()
            conflicting_ids = (
                gene_df[gene_df["paper_disease_category"] == "conflicting"]["paper_id"].str.lstrip("pmid:").tolist()
            )
            conflicting_counts = [
                json.loads(x)
                for x in gene_df[gene_df["paper_disease_category"] == "conflicting"][
                    "paper_disease_categorizations"
                ].tolist()
            ]

            paper_ids = gene_df["paper_id"].str.lstrip("pmid:").tolist()

            # Get the pmids from MGT.
            mgt_ids = mgt_df[mgt_df["gene"] == term]["pmid"].astype(str).tolist()

            # Cache the titles for these pmids.
            titles = get_paper_titles(set(paper_ids + mgt_ids))

            # Report the number of papers in each category
            f.write(f"Rare Disease Papers: {len(rare_disease_ids)}\n")
            benchmarking_results[term][0] = len(rare_disease_ids)
            f.write(f"Other Papers: {len(other_ids)}\n")
            benchmarking_results[term][1] = len(other_ids)
            f.write(f"Conflicting Papers: {len(conflicting_ids)}\n")
            benchmarking_results[term][2] = len(conflicting_ids)
            for i, id in enumerate(conflicting_ids):
                f.write(f"* {i + 1} * {id} * {titles[id]}\n")
                f.write(f"* {i + 1}-counts * {conflicting_counts[i]}\n")

            # If ev. agg. found rare disease papers, compare ev. agg. papers (PMIDs) to MGT data papers (PMIDs)
            print("Comparing Evidence Aggregator results to manual ground truth data for:", term, "...")

            # Calculate the number of correct, missed, and irrelevant papers for PubMed all up
            if args.isolated_run:
                p_corr, p_miss, p_irr = compare_pmid_lists(paper_ids, mgt_ids)

            # If the Evidence Aggregator classified rare disease papers, compare them to the MGT data papers
            if rare_disease_ids:
                correct_pmids, missed_pmids, irrelevant_pmids = compare_pmid_lists(rare_disease_ids, mgt_ids)

                # Report comparison between ev.agg. and MGT data
                f.write("\nOf Ev. Agg.'s rare disease papers...\n")
                f.write(f"E.A. # Correct Papers: {len(correct_pmids)}\n")
                f.write(f"E.A. # Missed Papers: {len(missed_pmids)}\n")
                f.write(f"E.A. # Irrelevant Papers: {len(irrelevant_pmids)}\n")

                # Calculate precision and recall
                if len(correct_pmids) > 0:
                    precision, recall, f1_score = calculate_metrics(correct_pmids, missed_pmids, irrelevant_pmids)

                    f.write(f"\nPrecision: {precision}\n")
                    f.write(f"Recall: {recall}\n")
                    f.write(f"F1 Score: {f1_score}\n")

                else:
                    f.write("\nNo true positives. Precision and recall are undefined.\n")

                # Update the metrics in the list for this gene
                benchmarking_results[term][3] = len(correct_pmids)
                benchmarking_results[term][4] = len(missed_pmids)
                benchmarking_results[term][5] = len(irrelevant_pmids)

                f.write(f"\nFound E.A. {len(correct_pmids)} correct.\n")
                for i, p in enumerate(correct_pmids):
                    f.write(f"* {i + 1} * {p} * {titles[p]}\n")  # PMID and title output

                f.write(f"\nFound E.A. {len(missed_pmids)} missed.\n")
                for i, p in enumerate(missed_pmids):
                    f.write(f"* {i + 1} * {p} * {titles[p]}\n")  # PMID and title output

                f.write(f"\nFound E.A. {len(irrelevant_pmids)} irrelevant.\n")
                for i, p in enumerate(irrelevant_pmids):
                    f.write(f"* {i + 1} * {p} * {titles[p]}\n")  # PMID and title output
                    # print(titles[p])

            else:
                f.write("\nOf Ev. Agg.'s rare disease papers...\n")
                f.write("E.A. # Correct Papers: 0\n")
                f.write("E.A. # Missed Papers: 0\n")
                f.write("E.A. # Irrelevant Papers: 0\n")

            # Compare PubMed papers to  MGT data papers
            if args.isolated_run:
                print("Comparing PubMed results to manual ground truth data for: ", term, "...")
                f.write("\nOf PubMed papers...\n")
                f.write(f"Pubmed # Correct Papers: {len(p_corr)}\n")
                f.write(f"Pubmed # Missed Papers: {len(p_miss)}\n")
                f.write(f"Pubmed # Irrelevant Papers: {len(p_irr)}\n")

                # Update the counts in the list for this gene
                benchmarking_results[term][6] = len(p_corr)
                benchmarking_results[term][7] = len(p_miss)
                benchmarking_results[term][8] = len(p_irr)

                f.write(f"\nFound Pubmed {len(p_corr)} correct.\n")
                for i, p in enumerate(p_corr):
                    f.write(f"* {i + 1} * {p} * {titles[p]}\n")  # PMID and title output

                f.write(f"\nFound Pubmed {len(p_miss)} missed.\n")
                for i, p in enumerate(p_miss):
                    f.write(f"* {i + 1} * {p} * {titles[p]}\n")  # PMID and title output

                f.write(f"\nFound Pubmed {len(p_irr)} irrelevant.\n")
                for i, p in enumerate(p_irr):
                    f.write(f"* {i + 1} * {p} * {titles[p]}\n")  # PMID and title output

    # Plot benchmarking results
    if args.isolated_run:
        results_to_plot = plot_benchmarking_results(benchmarking_results)
    else:
        results_to_plot = plot_benchmarking_results_tool_only(benchmarking_results)

    # Plot filtering results
    plot_filtered_categories(results_to_plot)

    print("Note: No need to view plots. They need to be updated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evidence Aggregator Paper Finding Benchmarks")
    parser.add_argument(
        "-l",
        "--library-config",
        nargs="?",
        default="lib/config/paper_finding_benchmark.yaml",
        type=str,
        help="Default is lib/config/paper_finding_benchmark.yaml",
    )
    parser.add_argument(
        "-m",
        "--mgt-train-test-path",
        nargs="?",
        default="data/v1/papers_train_v1.tsv",
        type=str,
        help="Default is data/v1/papers_train_v1.tsv",
    )
    parser.add_argument(
        "-p",
        "--pipeline-output",
        nargs="?",
        default=".out/library_benchmark.tsv",
        type=str,
        help="Default is .out/library_benchmark.tsv",
    )
    parser.add_argument(
        "-f",
        "--mgt-full-text-only",
        nargs="?",
        default=False,
        type=bool,
        help="Default is False",
    )
    parser.add_argument(
        "-t",
        "--plot-tool-only",
        nargs="?",
        default=False,
        type=bool,
        help="Default is False",
    )
    parser.add_argument(
        "-i",
        "--isolated-run",
        nargs="?",
        default=False,
        type=bool,
        help="Default is False. If True, this paper finding benchmarking run is considered to be an isolated benchmark."
        "Thus, paper finding benchmarks are run immediately. Not after entire an Evidence Aggregator pipeline run.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default=f".out/paper_finding_results_{(datetime.today().strftime('%Y-%m-%d'))}_{get_git_commit_hash()}",
        type=str,
        help=("Results output directory. Default is " f".out/paper_finding_results_<YYYY-MM-DD>_<GIT_COMMIT_HASH>/"),
    )
    args = parser.parse_args()

    print("Evidence Aggregator Paper Finding Benchmarks:")
    for arg, value in vars(args).items():
        print(f"- {arg}: {value}")

    print("\n")

    main(args)
