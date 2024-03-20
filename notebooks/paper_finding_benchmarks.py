"""Scipt to compare our Evidence Aggregator's paper finding pipeline to the manual ground truth (MGT)  data.

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

# Libraries
import argparse
import logging
import math
import os
import random
import warnings
from datetime import datetime
from typing import Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from lib.di import DiContainer
from lib.evagg.library import RareDiseaseFileLibrary
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import Paper

warnings.filterwarnings("ignore", category=DeprecationWarning)  # want to suppress pandas warning

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


def get_ground_truth_pmids(gene, mgt_gene_pmids_dict):
    """Get the manual ground truth (MGT)  data PMIDs for a gene.

    Return the PMIDs for the gene if it is in the ground truth  data, otherwise return None.
    """
    if gene in mgt_gene_pmids_dict.keys():
        return mgt_gene_pmids_dict[gene]
    else:
        return None


def compare_to_truth_or_tool(gene, input_papers, ncbi_lookup, mgt_gene_pmids_dict):
    """Compare the PMIDs that were found either by ev. agg. or competing too (e.g. PubMed) to the MGT PMIDs.

    Args:
        gene (str): the gene name
        input_papers (set): the papers that were found
        ncbi_lookup (IPaperLookupClient): the PubMed lookup client

    Returns:
        correct_pmids (list): the PMIDs and titles of the papers that were found that match the MGT data papers
        missed_pmids (list): the PMIDs and titles of the papers that are in the MGT data papers but not in the
        papers that were found
        irrelevant_pmids (list): the PMIDs and titles of the papers that are in the papers that were found but not in
        the MGT data papers
    """
    # Get all the PMIDs from all of the papers
    input_paper_pmids = [
        (getattr(paper, "props", {}).get("pmid", "Unknown"), getattr(paper, "props", {}).get("title", "Unknown"))
        for paper in input_papers
    ]

    ground_truth_gene_pmids = get_ground_truth_pmids(gene, mgt_gene_pmids_dict)

    # Keep track of the correct and extra PMIDs to subtract from the MGT data papers PMIDs
    counted_pmids = []
    correct_pmids = []
    missed_pmids = []
    irrelevant_pmids = []

    # For each gene, get MGT data PMIDs from ground_truth_papers_pmids and compare those PMIDS to the PMIDS
    # from the papers that the tool found (e.g. our ev. agg. tool or PubMed).
    if ground_truth_gene_pmids is not None:
        for pmid, title in input_paper_pmids:
            if pmid in ground_truth_gene_pmids:
                counted_pmids.append(pmid)
                correct_pmids.append((pmid, title))

            else:
                counted_pmids.append(pmid)
                irrelevant_pmids.append((pmid, title))

        # For any PMIDs in the MGT data that are not in the papers that were found, increment n_missed, use
        # counted_pmids to subtract from the MGT data papers PMIDs
        for pmid in ground_truth_gene_pmids:
            if pmid not in counted_pmids:
                missed_paper_title = ncbi_lookup.fetch(str(pmid))
                missed_paper_title = (
                    missed_paper_title.props.get("title", "Unknown") if missed_paper_title is not None else "Unknown"
                )
                missed_pmids.append((pmid, missed_paper_title))

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
    plt.title("Average number of correct, missed and extra papers for our tool and PubMed")

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


# Plot the filtereds into paper 3 categories
def plot_filtered_categories(results_to_plot):
    """Plot the filtered results (3 categories: rare disease, non-rare disease, other) in a barplot."""
    # Calculate averages
    rare_disease_avg = results_to_plot["Rare Disease Papers"].mean()
    non_rare_disease_avg = results_to_plot["Non-Rare Disease Papers"].mean()
    other_papers_avg = results_to_plot["Other Papers"].mean()

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
        [rare_disease_avg, non_rare_disease_avg, other_papers_avg],
        width=bar_width,
        alpha=0.5,
        color="b",
    )

    # Set the ticks to be first names
    plt.xticks(tick_pos, ["Rare Disease Papers", "Non-Rare Disease Papers", "Other Papers"])
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


def main(args):

    # Open and load the .yaml file
    with open(args.library_config, "r") as file:
        yaml_data = yaml.safe_load(file)

    # Read the manual ground truth (MGT) data from the CSV file
    mgt_gene_pmids_dict = read_mgt_split_tsv(args.mgt_train_test_path)

    # Get the query/ies from .yaml file
    query_list_yaml = []
    for query in yaml_data["queries"]:
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

    # Create the library and the PubMed lookup clients
    library: RareDiseaseFileLibrary = DiContainer().create_instance({"di_factory": "lib/config/library.yaml"}, {})
    ncbi_lookup: IPaperLookupClient = DiContainer().create_instance({"di_factory": "lib/config/ncbi_lookup.yaml"}, {})

    # Initialize the dictionary
    benchmarking_results = {}

    # For each query, get papers, compare ev. agg. papers to MGT data papers,
    # compare PubMed papers to MGT data papers. Write results to benchmarking against MGT file.
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "benchmarking_paper_finding_results_train.txt"), "w") as f:
        for query in query_list_yaml:

            # Get the gene name from the query
            term = query["gene_symbol"]
            f.write(f"\nGENE: {term}\n")
            print("Finding papers for: ", query["gene_symbol"], "...")

            # Initialize the list for this gene
            benchmarking_results[term] = [0] * 9

            # Get the papers from the library for the query (i.e. gene/term)
            rare_disease_papers, non_rare_disease_papers, other_papers, papers = library.get_all_papers(
                query
            )  # TODO: better to union 3 sets or keep papers?

            # Check if =<3 papers categories are empty, and report the number of papers in each category
            if rare_disease_papers == Set[Paper]:
                f.write(f"Rare Disease Papers: {0}\n")
                benchmarking_results[term][0] = 0
            else:
                f.write(f"Rare Disease Papers: {len(rare_disease_papers)}\n")
                benchmarking_results[term][0] = len(rare_disease_papers)
            if non_rare_disease_papers == Set[Paper]:
                f.write(f"Non-Rare Disease Papers: {0}\n")
                benchmarking_results[term][1] = 0
            else:
                f.write(f"Non-Rare Disease Papers: {len(non_rare_disease_papers)}\n")
                benchmarking_results[term][1] = len(non_rare_disease_papers)
            if other_papers == Set[Paper]:
                f.write(f"Other Papers: {0}\n")
                benchmarking_results[term][2] = 0
            else:
                f.write(f"Other Papers: {len(other_papers)}\n")
                benchmarking_results[term][2] = len(other_papers)

            # If ev. agg. found rare disease papers, compare ev. agg. papers (PMIDs) to MGT data papers (PMIDs)
            print("Comparing Evidence Aggregator results to manual ground truth data for:", query["gene_symbol"], "...")
            if rare_disease_papers != Set[Paper]:
                correct_pmids, missed_pmids, irrelevant_pmids = compare_to_truth_or_tool(
                    term, rare_disease_papers, ncbi_lookup, mgt_gene_pmids_dict
                )

                # Report comparison between ev.agg. and MGT data
                f.write("\nOf Ev. Agg.'s rare disease papers...\n")
                f.write(f"E.A. # Correct Papers: {len(correct_pmids)}\n")
                f.write(f"E.A. # Missed Papers: {len(missed_pmids)}\n")
                f.write(f"E.A. # Irrelevant Papers: {len(irrelevant_pmids)}\n")

                # Update the metrics in the list for this gene
                benchmarking_results[term][3] = len(correct_pmids)
                benchmarking_results[term][4] = len(missed_pmids)
                benchmarking_results[term][5] = len(irrelevant_pmids)

                f.write(f"\nFound E.A. {len(correct_pmids)} correct.\n")
                for i, p in enumerate(correct_pmids):
                    f.write(f"* {i + 1} * {p[0]} * {p[1]}\n")  # PMID and title output

                f.write(f"\nFound E.A. {len(missed_pmids)} missed.\n")
                for i, p in enumerate(missed_pmids):
                    f.write(f"* {i + 1} * {p[0]} * {p[1]}\n")  # PMID and title output

                f.write(f"\nFound E.A. {len(irrelevant_pmids)} irrelevant.\n")
                for i, p in enumerate(irrelevant_pmids):
                    f.write(f"* {i + 1} * {p[0]} * {p[1]}\n")  # PMID and title output
            else:
                f.write("\nOf Ev. Agg.'s rare disease papers...\n")
                f.write(f"E.A. # Correct Papers: {0}\n")
                f.write(f"E.A. # Missed Papers: {0}\n")
                f.write(f"E.A. # Irrelevant Papers: {0}\n")

            # Compare PubMed papers to  MGT data papers
            print("Comparing PubMed results to manual ground truth data for: ", query["gene_symbol"], "...")
            p_corr, p_miss, p_irr = compare_to_truth_or_tool(term, papers, ncbi_lookup, mgt_gene_pmids_dict)
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
                f.write(f"* {i + 1} * {p[0]} * {p[1]}\n")  # PMID and title output

            f.write(f"\nFound Pubmed {len(p_miss)} missed.\n")
            for i, p in enumerate(p_miss):
                f.write(f"* {i + 1} * {p[0]} * {p[1]}\n")  # PMID and title output

            f.write(f"\nFound Pubmed {len(p_irr)} irrelevant.\n")
            for i, p in enumerate(p_irr):
                f.write(f"* {i + 1} * {p[0]} * {p[1]}\n")  # PMID and title output

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
        default="lib/config/pubmed_library_config.yaml",
        type=str,
        help="Default is lib/config/pubmed_library_config.yaml",
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
