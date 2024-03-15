# paper_finding_benchmarks.py
"""Scipt to compare our Evidence Aggregator's paper finding pipeline to the manual ground truth (MGT) training data"""
# Inputs
# - MGT data
# Outputs:
#
# Libraries
import argparse
import datetime
import logging
import math
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)  # want to suppress pandas warning

from datetime import datetime
from typing import Set

import pandas as pd
import yaml

from lib.di import DiContainer
from lib.evagg.library import RareDiseaseFileLibrary
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import Paper

logger = logging.getLogger(__name__)


# Set the logging level to CRITICAL
def set_log_level():
    logging.getLogger().setLevel(logging.CRITICAL)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    for name, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):  # Just to be sure it is a Logger object
            logger.setLevel(logging.CRITICAL)


# Function to read the manual ground truth (MGT) data from the CSV file
def read_mgt(mgt_paper_finding_path):
    """Read the manual ground truth (MGT) data from the CSV file.
    Return a dictionary with the gene names and their corresponding PMIDs."""

    mgt_gene_pmids_df = pd.read_csv(mgt_paper_finding_path)

    # Initialize dictionary to store the gene names and their corresponding PMIDs
    mgt_gene_pmids_dict = {}

    # Iterate over the rows and parse the ones we care about
    for index in range(len(mgt_gene_pmids_df) - 1):
        # Get the gene name
        gene = mgt_gene_pmids_df.iloc[index]["gene"]

        # Skip rows without a gene name (e.g. notes or process)
        if pd.isna(gene):
            continue

        # Get the PMIDs
        pmids = mgt_gene_pmids_df.iloc[index, 5:].dropna().tolist()

        # Get the exclusion marks (to exclude PMIDs that are marked with 'x', i.e. not relevant for V1)
        marks = mgt_gene_pmids_df.iloc[index + 1, 5:].tolist()

        # Exclude the PMIDs that are marked with 'x'
        pmids = [pmid for pmid, mark in zip(pmids, marks) if mark != "x"]

        # Add the gene and PMIDs to the dictionary
        mgt_gene_pmids_dict[gene] = pmids

    # Get the first key-value pair
    first_key = list(mgt_gene_pmids_dict.keys())[0]
    first_value = mgt_gene_pmids_dict[first_key]

    logger.info(f"Structure of manual ground truth data dataframe:\n {first_key}: {first_value}\n ...")
    return mgt_gene_pmids_dict


# Function to split genes into train 70% and test 30%
def create_train_test_split(mgt_gene_pmids_dict):
    """Split the genes into train 70% and test 30%.
    Return the train and test dictionaries of gene names and their corresponding PMIDs."""

    # Get the list of genes
    genes = sorted(list(mgt_gene_pmids_dict.keys()))

    # Shuffle the list
    random.seed(0)
    random.shuffle(genes)

    # Calculate the index for splitting
    split_index = math.ceil(0.7 * len(genes))

    # Split the list into train and test
    train = genes[:split_index]
    test = genes[split_index:]

    logger.info("Training set of genes:", len(train), train)
    logger.info("Testing set of genes:", len(test), test)

    # Filter mgt_paper_finding_dict to only include genes in train
    train_mgt_paper_finding_dict = {gene: mgt_gene_pmids_dict[gene] for gene in train}

    # Filter mgt_paper_finding_dict to only include genes in train
    test_mgt_paper_finding_dict = {gene: mgt_gene_pmids_dict[gene] for gene in test}

    return train_mgt_paper_finding_dict, test_mgt_paper_finding_dict


# Get the manual ground truth (MGT) training data PMIDs for a gene.
def get_ground_truth_pmids(gene, train_mgt_paper_finding_dict):
    """Get the manual ground truth (MGT) training data PMIDs for a gene.
    Return the PMIDs for the gene if it is in the ground truth training data, otherwise return None."""

    if gene in train_mgt_paper_finding_dict.keys():
        return train_mgt_paper_finding_dict[gene]
    else:
        return None


# Compare the papers that were found to the manual ground truth (MGT) training data papers
def compare_to_truth_or_tool(gene, input_papers, is_pubmed, ncbi_lookup, train_mgt_paper_finding_dict):
    """Compare the PMIDs that were found either by ev. agg. or competing too (e.g. PubMed) to the MGT PMIDs.
    Args:
        gene (str): the gene name
        input_papers (set): the papers that were found
        is_pubmed (int): 1 if comparing to PubMed results, 0 if comparing to our E.A. tool
        ncbi_lookup (IPaperLookupClient): the PubMed lookup client
    Returns:
        correct_pmids (list): the PMIDs and titles of the papers that were found that match the MGT training data papers
        missed_pmids (list): the PMIDs and titles of the papers that are in the MGT training data papers but not in the
        papers that were found
        irrelevant_pmids (list): the PMIDs and titles of the papers that are in the papers that were found but not in
        the MGT training data papers
    """

    # Get all the PMIDs from all of the papers
    # r_d_pmids = [(paper.props.get("pmid", "Unknown"), paper.props.get("title", "Unknown")) for paper in r_d_papers]
    input_paper_pmids = [
        (getattr(paper, "props", {}).get("pmid", "Unknown"), getattr(paper, "props", {}).get("title", "Unknown"))
        for paper in input_papers
    ]

    ground_truth_gene_pmids = get_ground_truth_pmids(gene, train_mgt_paper_finding_dict)

    # Keep track of the correct and extra PMIDs to subtract from the MGT training data papers PMIDs
    counted_pmids = []
    correct_pmids = []
    missed_pmids = []
    irrelevant_pmids = []

    # For each gene, get MGT training data PMIDs from ground_truth_papers_pmids and compare those PMIDS to the PMIDS
    # from the papers that the tool found (e.g. our ev. agg. tool or PubMed).
    if ground_truth_gene_pmids is not None:
        for pmid, title in input_paper_pmids:
            if pmid in ground_truth_gene_pmids:
                counted_pmids.append(pmid)
                correct_pmids.append((pmid, title))

            else:
                counted_pmids.append(pmid)
                irrelevant_pmids.append((pmid, title))

        # For any PMIDs in the MGT training data that are not in the papers that were found, increment n_missed, use
        # counted_pmids to subtract from the MGT training data papers PMIDs
        for pmid in ground_truth_gene_pmids:
            if pmid not in counted_pmids:
                missed_paper_title = ncbi_lookup.fetch(str(pmid))
                missed_paper_title = (
                    missed_paper_title.props.get("title", "Unknown") if missed_paper_title is not None else "Unknown"
                )
                missed_pmids.append((pmid, missed_paper_title))

    return correct_pmids, missed_pmids, irrelevant_pmids


# parse benchmarking_paper_finding_results_train.txt to plot the results
def parse_benchmarking_results(file_path):
    # Parse benchmarking results from its analyst readable format to a dictionary
    benchmarking_train_results = {}
    with open(file_path, "r") as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("GENE:"):
                gene = line.split()[1]
                i += 1
                while i < len(lines) and not lines[i].startswith("GENE:"):
                    if lines[i].startswith("Rare Disease Papers:"):
                        rare_disease_papers = int(lines[i].split(":")[1].strip())
                    elif lines[i].startswith("Non-Rare Disease Papers:"):
                        non_rare_disease_papers = int(lines[i].split(":")[1].strip())
                    elif lines[i].startswith("Other Papers:"):
                        other_papers = int(lines[i].split(":")[1].strip())
                    elif lines[i].startswith("E.A. # Correct Papers:"):
                        tool_correct_papers = int(lines[i].split(":")[1].strip())
                    elif lines[i].startswith("E.A. # Missed Papers:"):
                        tool_missed_papers = int(lines[i].split(":")[1].strip())
                    elif lines[i].startswith("E.A. # Irrelevant Papers:"):
                        tool_irrelevant_papers = int(lines[i].split(":")[1].strip())
                    elif lines[i].startswith("Pubmed # Correct Papers:"):
                        pubmed_correct_papers = int(lines[i].split(":")[1].strip())
                    elif lines[i].startswith("Pubmed # Missed Papers:"):
                        pubmed_missed_papers = int(lines[i].split(":")[1].strip())
                    elif lines[i].startswith("Pubmed # Irrelevant Papers:"):
                        pubmed_irrelevant_papers = int(lines[i].split(":")[1].strip())
                    i += 1
                if "rare_disease_papers" not in locals():
                    rare_disease_papers = 0
                if "non_rare_disease_papers" not in locals():
                    non_rare_disease_papers = 0
                if "other_papers" not in locals():
                    other_papers = 0
                if "tool_correct_papers" not in locals():
                    tool_correct_papers = 0
                if "tool_missed_papers" not in locals():
                    tool_missed_papers = 0
                if "tool_irrelevant_papers" not in locals():
                    tool_irrelevant_papers = 0
                if "pubmed_correct_papers" not in locals():
                    pubmed_correct_papers = 0
                if "pubmed_missed_papers" not in locals():
                    pubmed_missed_papers = 0
                if "pubmed_irrelevant_papers" not in locals():
                    pubmed_irrelevant_papers = 0

                benchmarking_train_results[gene] = [
                    rare_disease_papers,
                    non_rare_disease_papers,
                    other_papers,
                    tool_correct_papers,
                    tool_missed_papers,
                    tool_irrelevant_papers,
                    pubmed_correct_papers,
                    pubmed_missed_papers,
                    pubmed_irrelevant_papers,
                ]
            else:
                i += 1  # Go to the next line

    return benchmarking_train_results


def plot_benchmarking_results(benchmarking_train_results):

    # Python
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
        # using the 'Tool' data
        tool_averages,
        # set the width
        width=bar_width,
        # with the label 'Tool'
        label="Tool",
        # with alpha 0.5
        alpha=0.5,
        # with color
        color="b",
        # with error
        yerr=tool_std,
    )

    # Create a bar plot for 'PubMed'
    pubmed_bars = ax.bar(
        bar_l + bar_width,
        # using the 'PubMed' data
        pubmed_averages,
        # set the width
        width=bar_width,
        # with the label 'PubMed'
        label="PubMed",
        # with alpha 0.5
        alpha=0.5,
        # with color
        color="r",
        # with error
        yerr=pubmed_std,
    )

    # Set the ticks to be first names
    plt.xticks(tick_pos, tool_averages.index)
    ax.set_ylabel("Average")
    ax.set_xlabel("Categories")
    plt.legend(loc="upper right")
    plt.title("Average number of correct, missed and extra papers for our tool and PubMed")

    # Function to add labels
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


# Plot the filtereds into paper 3 categories (rare disease, non-rare disease, other)
def plot_paper_categories(rare_disease_papers, non_rare_disease_papers, other_papers, gene):

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
        # using the data
        [rare_disease_avg, non_rare_disease_avg, other_papers_avg],
        # set the width
        width=bar_width,
        # with alpha 0.5
        alpha=0.5,
        # with color
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

    # Let's display the plot
    plt.show()


def main(args):

    # Set the logging level
    set_log_level()  # TODO: set_log_level() to something that can be configured from CRITICAL

    # Open and load the .yaml file
    with open(args.library_config, "r") as file:
        yaml_data = yaml.safe_load(file)

    # Get max_papers parameter
    max_papers = yaml_data["library"]["max_papers"]  # TODO: use this parameter to limit the number of papers to fetch

    # Read the manual ground truth (MGT) data from the CSV file
    mgt_gene_pmids_dict = read_mgt(args.mgt_path)

    # Create train and test split from MGT data
    train_mgt_paper_finding_dict, _ = create_train_test_split(
        mgt_gene_pmids_dict
    )  # TODO: fill in _ with test data benchmarking, or reconfigure to accept train or test data

    # Get the query/ies from .yaml file
    query_list_yaml = []
    for query in yaml_data["queries"]:
        # Extract the values, or use an empty string if they're missing
        gene_symbol = query.get("gene_symbol", "")
        min_date = query.get("min_date", "")
        max_date = query.get("max_date", "")
        date_type = query.get("date_type", "")

        # Create a new dictionary with the gene_symbol and the values, '' in places where .yaml does not show a value
        new_dict = {"gene_symbol": gene_symbol, "min_date": min_date, "max_date": max_date, "date_type": date_type}

        # Add the new dictionary to the list
        query_list_yaml.append(new_dict)

    # Create the library and the PubMed lookup clients
    library: RareDiseaseFileLibrary = DiContainer().create_instance({"di_factory": "lib/config/library.yaml"}, {})
    ncbi_lookup: IPaperLookupClient = DiContainer().create_instance({"di_factory": "lib/config/ncbi_lookup.yaml"}, {})

    # For each query, get papers, compare ev. agg. papers to MGT training data papers,
    # compare PubMed papers to MGT training data papers. Write results to benchmarking against training set file.
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "benchmarking_paper_finding_results_train.txt"), "w") as f:
        for query in query_list_yaml:

            # Get the gene name from the query
            term = query["gene_symbol"]
            f.write("\nGENE: %s\n" % term)

            # Get the papers from the library for the query (i.e. gene/term)
            rare_disease_papers, count_r_d_papers, non_rare_disease_papers, other_papers, papers = library.get_papers(
                query
            )  # TODO: remove count_r_d_papers, better to union 3 sets or keep papers?

            # Check if =<3 papers categories are empty, and report the number of papers in each category
            if rare_disease_papers == Set[Paper]:
                f.write("Rare Disease Papers: 0\n")
            else:
                f.write("Rare Disease Papers: %s\n" % len(rare_disease_papers))  # type: ignore
            if non_rare_disease_papers == Set[Paper]:
                f.write("Non-Rare Disease Papers: 0\n")
            else:
                f.write("Non-Rare Disease Papers: %s\n" % len(non_rare_disease_papers))  # type: ignore
            if other_papers == Set[Paper]:
                f.write("Other Papers: 0\n")
            else:
                f.write("Other Papers: %s\n" % len(other_papers))

            # If ev. agg. found rare disease papers, compare ev. agg. papers (PMIDs) to the MGT training data papers (PMIDs)
            if count_r_d_papers != 0:
                correct_pmids, missed_pmids, irrelevant_pmids = compare_to_truth_or_tool(
                    term, rare_disease_papers, 0, ncbi_lookup, train_mgt_paper_finding_dict
                )

                # Report comparison between ev.agg. and MGT training data
                f.write("\nOf Ev. Agg.'s rare disease papers...\n")
                f.write(f"E.A. # Correct Papers: {len(correct_pmids)}\n")
                f.write(f"E.A. # Missed Papers: {len(missed_pmids)}\n")
                f.write(f"E.A. # Irrelevant Papers: {len(irrelevant_pmids)}\n")

                f.write(f"\nFound E.A. {len(correct_pmids)} correct.\n")
                for i, p in enumerate(correct_pmids):
                    f.write("* %s * %s * %s\n" % (i + 1, p[0], p[1]))  # PMID and title output

                f.write(f"\nFound E.A. {len(missed_pmids)} missed.\n")
                for i, p in enumerate(missed_pmids):
                    f.write("* %s * %s * %s\n" % (i + 1, p[0], p[1]))  # PMID and title output

                f.write(f"\nFound E.A. {len(irrelevant_pmids)} irrelevant.\n")
                for i, p in enumerate(irrelevant_pmids):
                    f.write("* %s * %s * %s\n" % (i + 1, p[0], p[1]))  # PMID and title output
            else:
                f.write("\nOf Ev. Agg.'s rare disease papers...\n")
                f.write(f"E.A. # Correct Papers: 0\n")
                f.write(f"E.A. # Missed Papers: 0\n")
                f.write(f"E.A. # Irrelevant Papers: 0\n")

            # Compare PubMed papers to  MGT training data papers
            p_corr, p_miss, p_irr = compare_to_truth_or_tool(term, papers, 1, ncbi_lookup, train_mgt_paper_finding_dict)
            f.write("\nOf PubMed papers...\n")
            f.write(f"Pubmed # Correct Papers: {len(p_corr)}\n")
            f.write(f"Pubmed # Missed Papers: {len(p_miss)}\n")
            f.write(f"Pubmed # Irrelevant Papers: {len(p_irr)}\n")

            f.write(f"\nFound Pubmed {len(p_corr)} correct.\n")
            for i, p in enumerate(p_corr):
                f.write("* %s * %s * %s\n" % (i + 1, p[0], p[1]))  # TODO: update with f string # PMID and title output

            f.write(f"\nFound Pubmed {len(p_miss)} missed.\n")
            for i, p in enumerate(p_miss):
                f.write("* %s * %s * %s\n" % (i + 1, p[0], p[1]))  # PMID and title output

            f.write(f"\nFound Pubmed {len(p_irr)} irrelevant.\n")
            for i, p in enumerate(p_irr):
                f.write("* %s * %s * %s\n" % (i + 1, p[0], p[1]))  # PMID and title output

    # Parse benchmarking results from file (its analyst readable format) and plot them
    benchmark_file = os.path.join(args.outdir, "benchmarking_paper_finding_results_train.txt")
    benchmarking_train_results = parse_benchmarking_results(benchmark_file)
    plot_benchmarking_results(benchmarking_train_results)


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
        "--mgt-path",
        nargs="?",
        default="data/Manual_Ground_Truth_find_right_papers.csv",
        type=str,
        help="Default is data/Manual_Ground_Truth_find_right_papers.csv",
    )
    parser.add_argument("--log_level", default="CRITICAL", type=str, help="Default is CRITICAL")
    parser.add_argument(
        "--outdir",
        default=".out/paper_finding_results_%s"
        % (datetime.today().strftime("%Y-%m-%d")),  # TODO: consider more granular date format
        type=str,
        help="Results output directory. Default is .out/paper_finding_results_%s/"
        % (datetime.today().strftime("%Y-%m-%d")),
    )
    args = parser.parse_args()

    main(args)
