"""Script to compare our Evidence Aggregator's paper finding pipeline to the manual ground truth (MGT)  data.

Inputs
- MGT intermediate/train-test split data file (e.g. data/v1/papers_train_v1.tsv)
- .yaml of queries

Outputs:
- benchmarking_paper_finding_results_train.txt: comparison of the papers that were found to the MGT data
papers
- benchmarking_paper_finding_results_train.png: barplot of average number of correct, missed, and irrelevant papers
for the Evidence Aggregator tool and PubMed
- filtering_paper_finding_results_train.png: barplot of average number of papers filtered into rare, non-rare, and
other categories
"""

import argparse
import logging
import os

# Libraries
import warnings
from datetime import datetime
from functools import cache
from typing import Dict, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from lib.di import DiContainer
from lib.evagg.ref import IPaperLookupClient

# warnings.filterwarnings("ignore", category=DeprecationWarning)  # want to suppress pandas warning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_mgt_split_tsv(mgt_split_tsv):
    """Build a gene:["PMID_1, "PMID_2", ...] dictionary from an input tsv file.

    Returns a dictionary of gene:["PMID_1, "PMID_2", ...].
    """
    mgt_paper_finding_dict = {}
    with open(mgt_split_tsv, "r") as file:
        next(file)  # Skip the header
        for line in file:
            gene, pmid, _ = line.strip().split("\t")
            if gene in mgt_paper_finding_dict:
                mgt_paper_finding_dict[gene].append(pmid)
            else:
                mgt_paper_finding_dict[gene] = [pmid]
    return mgt_paper_finding_dict


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
    correct_pmids = list(set(input_pmids).intersection(truth_pmids))
    missed_pmids = list(set(truth_pmids).difference(input_pmids))
    irrelevant_pmids = list(set(input_pmids).difference(truth_pmids))

    return correct_pmids, missed_pmids, irrelevant_pmids


def plot_benchmarking_results(benchmarking_train_results):
    """Plot the benchmarking results from the  set in a barplot.

    Return the results dataframe (see keys below) for subsequent plotting.
    """
    keys = [
        "Gene",
        "Rare Disease Papers",
        "Non-Rare Disease Papers",
        "Other Papers",
        "Discordant Papers",
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
    """Plot the filtered results (3 categories: rare disease, non-rare disease, other) in a barplot."""
    # Calculate averages
    rare_disease_avg = results_to_plot["Rare Disease Papers"].mean()
    non_rare_disease_avg = results_to_plot["Non-Rare Disease Papers"].mean()
    other_papers_avg = results_to_plot["Other Papers"].mean()
    discordant_papers_avg = results_to_plot["Discordant Papers"].mean()

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # Define bar width
    bar_width = 0.35

    # Positions of the left bar boundaries
    bar_l = np.arange(4)

    # Positions of the x-axis ticks (center of the bars as bar labels)
    tick_pos = [i + (bar_width / 2) for i in bar_l]

    # Create a bar plot
    bars = ax.bar(
        bar_l,
        [rare_disease_avg, non_rare_disease_avg, other_papers_avg, discordant_papers_avg],
        width=bar_width,
        alpha=0.5,
        color="b",
    )

    # Set the ticks to be first names
    plt.xticks(
        tick_pos, ["Rare Disease Papers", "Non-Rare Disease Papers", "Other Papers", "Discordant Papers"], rotation=10
    )
    ax.set_ylabel("Average")
    ax.set_xlabel("Categories")
    plt.title("Average # PubMed papers filtered into rare, non-rare, and other categories")

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

    if any(x not in yaml_genes for x in pipeline_df.gene.unique().tolist()):
        raise ValueError("Gene(s) in pipeline output not found in the .yaml file.")

    # Initialize the dictionary
    benchmarking_results = {}

    # Average precision and recall for all genes
    avg_precision = []
    avg_recall = []

    # For each query, get papers, compare ev. agg. papers to MGT data papers,
    # compare PubMed papers to MGT data papers. Write results to benchmarking against MGT file.
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "benchmarking_paper_finding_results_train.txt"), "w") as f:
        for gene_symbol, gene_df in pipeline_df.groupby("gene"):
            # Get the gene name from the query
            term = gene_symbol
            f.write(f"\nGENE: {term}\n")
            print("Analyzing found papers for: ", gene_symbol, "...")

            # Initialize the list for this gene
            benchmarking_results[term] = [0] * 10

            # Get pmids from pipeline output for this gene.
            rare_disease_ids = (
                gene_df[gene_df["paper_disease_category"] == "rare disease"]["paper_id"].str.lstrip("pmid:").tolist()
            )
            non_rare_disease_ids = (
                gene_df[gene_df["paper_disease_category"] == "non-rare disease"]["paper_id"]
                .str.lstrip("pmid:")
                .tolist()
            )
            other_ids = gene_df[gene_df["paper_disease_category"] == "other"]["paper_id"].str.lstrip("pmid:").tolist()
            discordant_ids = (
                gene_df[gene_df["paper_disease_category"] == "discordant"]["paper_id"].str.lstrip("pmid:").tolist()
            )
            discordant_counts = gene_df[gene_df["paper_disease_category"] == "discordant"][
                "paper_disease_categorizations"
            ].tolist()

            paper_ids = gene_df["paper_id"].str.lstrip("pmid:").tolist()

            # Get the pmids from MGT.
            mgt_ids = mgt_df[mgt_df["gene"] == term]["pmid"].astype(str).tolist()

            # Cache the titles for these pmids.
            titles = get_paper_titles(set(paper_ids + mgt_ids))

            # Report the number of papers in each category
            f.write(f"Rare Disease Papers: {len(rare_disease_ids)}\n")
            benchmarking_results[term][0] = len(rare_disease_ids)
            f.write(f"Non-Rare Disease Papers: {len(non_rare_disease_ids)}\n")
            benchmarking_results[term][1] = len(non_rare_disease_ids)
            f.write(f"Other Papers: {len(other_ids)}\n")
            benchmarking_results[term][2] = len(other_ids)
            f.write(f"Discordant Papers: {len(discordant_ids)}\n")
            benchmarking_results[term][3] = len(discordant_ids)
            for i, id in enumerate(discordant_ids):
                f.write(f"* {i + 1} * {id} * {titles[id]}\n")
                f.write(f"* {i + 1}-counts * {discordant_counts[i]}\n")

            # If ev. agg. found rare disease papers, compare ev. agg. papers (PMIDs) to MGT data papers (PMIDs)
            print("Comparing Evidence Aggregator results to manual ground truth data for:", gene_symbol, "...")
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
                    avg_precision.append(precision)
                    avg_recall.append(recall)
                else:
                    f.write("\nNo true positives. Precision and recall are undefined.\n")

                # Update the metrics in the list for this gene
                benchmarking_results[term][4] = len(correct_pmids)
                benchmarking_results[term][5] = len(missed_pmids)
                benchmarking_results[term][6] = len(irrelevant_pmids)

                f.write(f"\nFound E.A. {len(correct_pmids)} correct.\n")
                for i, p in enumerate(correct_pmids):
                    f.write(f"* {i + 1} * {p} * {titles[p]}\n")  # PMID and title output

                f.write(f"\nFound E.A. {len(missed_pmids)} missed.\n")
                for i, p in enumerate(missed_pmids):
                    f.write(f"* {i + 1} * {p} * {titles[p]}\n")  # PMID and title output

                f.write(f"\nFound E.A. {len(irrelevant_pmids)} irrelevant.\n")
                for i, p in enumerate(irrelevant_pmids):
                    f.write(f"* {i + 1} * {p} * {titles[p]}\n")  # PMID and title output

            else:
                f.write("\nOf Ev. Agg.'s rare disease papers...\n")
                f.write("E.A. # Correct Papers: 0\n")
                f.write("E.A. # Missed Papers: 0\n")
                f.write("E.A. # Irrelevant Papers: 0\n")

            # Compare PubMed papers to  MGT data papers
            print("Comparing PubMed results to manual ground truth data for: ", gene_symbol, "...")
            p_corr, p_miss, p_irr = compare_pmid_lists(paper_ids, mgt_ids)
            f.write("\nOf PubMed papers...\n")
            f.write(f"Pubmed # Correct Papers: {len(p_corr)}\n")
            f.write(f"Pubmed # Missed Papers: {len(p_miss)}\n")
            f.write(f"Pubmed # Irrelevant Papers: {len(p_irr)}\n")

            # Update the counts in the list for this gene
            benchmarking_results[term][7] = len(p_corr)
            benchmarking_results[term][8] = len(p_miss)
            benchmarking_results[term][9] = len(p_irr)

            f.write(f"\nFound Pubmed {len(p_corr)} correct.\n")
            for i, p in enumerate(p_corr):
                f.write(f"* {i + 1} * {p} * {titles[p]}\n")  # PMID and title output

            f.write(f"\nFound Pubmed {len(p_miss)} missed.\n")
            for i, p in enumerate(p_miss):
                f.write(f"* {i + 1} * {p} * {titles[p]}\n")  # PMID and title output

            f.write(f"\nFound Pubmed {len(p_irr)} irrelevant.\n")
            for i, p in enumerate(p_irr):
                f.write(f"* {i + 1} * {p} * {titles[p]}\n")  # PMID and title output

        # Calculate average precision and recall
        if len(avg_precision) != 0:
            avg_precision = sum(avg_precision) / len(avg_precision)
            avg_recall = sum(avg_recall) / len(avg_recall)

            # Write average precision and recall to the file
            f.write(f"\nAverage Precision: {avg_precision}\n")
            f.write(f"Average Recall: {avg_recall}\n")
        else:
            f.write("\nNo true positives. Precision and recall are undefined.\n")

    # Plot benchmarking results
    results_to_plot = plot_benchmarking_results(benchmarking_results)

    # Plot filtering results
    plot_filtered_categories(results_to_plot)


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
        "--outdir",
        default=f".out/paper_finding_results_{(datetime.today().strftime('%Y-%m-%d'))}",
        type=str,
        help=(
            "Results output directory. Default is "
            f".out/paper_finding_results_{(datetime.today().strftime('%Y-%m-%d'))}/"
        ),
    )
    args = parser.parse_args()

    main(args)
