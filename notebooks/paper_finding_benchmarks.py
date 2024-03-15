# Compare our tool's paper finding pipeline to the manual ground truth training for rare disease papers
import logging
import math
import random

import numpy as np
import pandas as pd
import pytest
import yaml

from lib.di import DiContainer
from lib.evagg.interfaces import IGetPapers
from lib.evagg.library import RareDiseaseFileLibrary
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import Paper

# Set path for manual ground truth (MGT) data CSV file

mgt_paper_finding_path = "/home/azureuser/ev-agg-exp/data/Manual_Ground_Truth_find_right_papers.csv"


# %%
# Read the manual ground truth (MGT) data from the CSV file

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

print(f"Structure of manual ground truth data dataframe:\n {first_key}: {first_value}\n ...")
# %%


# %%
# Split genes into train 70% and test 30%

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

print("Train:", len(train), train)
print("Test:", len(test), test)

# Filter mgt_paper_finding_dict to only include genes in train
train_mgt_paper_finding_dict = {gene: mgt_gene_pmids_dict[gene] for gene in train}

# Filter mgt_paper_finding_dict to only include genes in train
test_mgt_paper_finding_dict = {gene: mgt_gene_pmids_dict[gene] for gene in test}
# %%


# %%
# Get the manual ground truth (MGT) training data PMIDs for a gene.
def get_ground_truth_pmids(gene, train_mgt_paper_finding_dict):
    if gene in train_mgt_paper_finding_dict.keys():
        return train_mgt_paper_finding_dict[gene]
    else:
        print(gene, "not in ground truth training data")
        return None


# Compare the papers that were found to the manual ground truth (MGT) training data papers
def compare_to_truth_or_tool(gene, input_papers, is_pubmed, ncbi_lookup):
    """Compare the papers that were found to the ground truth papers.
    Args:
        paper (Paper): The paper to compare.
    Returns:
        number of correct papers (i.e. the number that match the MGT training data papers)
        number of missed papers (i.e. the number that are in the MGT training data papers but not in the papers that were found)
        number of extra papers (i.e. the number that are in the papers that were found but not in the MGT training data papers)
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

    # For each gene, get MGT training data PMIDs from ground_truth_papers_pmids and compare those PMIDS to the PMIDS from the papers that the tool found (e.g. our ev. agg. tool or PubMed).
    if ground_truth_gene_pmids is not None:
        for pmid, title in input_paper_pmids:
            if pmid in ground_truth_gene_pmids:
                counted_pmids.append(pmid)
                correct_pmids.append((pmid, title))

            else:
                counted_pmids.append(pmid)
                irrelevant_pmids.append((pmid, title))

        # For any PMIDs in the MGT training data that are not in the papers that were found, increment n_missed, use counted_pmids to subtract from the MGT training data papers PMIDs
        for pmid in ground_truth_gene_pmids:
            if pmid not in counted_pmids:
                missed_paper_title = ncbi_lookup.fetch(str(pmid))
                missed_paper_title = (
                    missed_paper_title.props.get("title", "Unknown") if missed_paper_title is not None else "Unknown"
                )
                missed_pmids.append((pmid, missed_paper_title))

    if is_pubmed:  # comparing PubMed results to MGT training data
        print("\nOf PubMed papers...")
        print("Pubmed # Correct Papers: ", len(correct_pmids))
        print("Pubmed # Missed Papers: ", len(missed_pmids))
        print("Pubmed # Irrelevant Papers: ", len(irrelevant_pmids))

        print(f"\nFound Pubmed {len(correct_pmids)} correct.")
        for i, p in enumerate(correct_pmids):
            print("*", i + 1, "*", p[0], "*", p[1])

        print(f"\nFound Pubmed {len(missed_pmids)} missed.")
        for i, p in enumerate(missed_pmids):
            print("*", i + 1, "*", p[0], "*", p[1])

        print(f"\nFound Pubmed {len(irrelevant_pmids)} irrelevant.")
        for i, p in enumerate(irrelevant_pmids):
            print("*", i + 1, "*", p[0], "*", p[1])
    else:  # Comparing tool to manual ground truth (MGT) training data
        print("\nOf the rare disease papers...")
        print("Tool # Correct Papers: ", len(correct_pmids))
        print("Tool # Missed Papers: ", len(missed_pmids))
        print("Tool # Irrelevant Papers: ", len(irrelevant_pmids))

        print(f"\nFound tool {len(correct_pmids)} correct.")
        for i, p in enumerate(correct_pmids):
            print("*", i + 1, "*", p[0], "*", p[1])

        print(f"\nFound tool {len(missed_pmids)} missed.")
        for i, p in enumerate(missed_pmids):
            print("*", i + 1, "*", p[0], "*", p[1])

        print(f"\nFound tool {len(irrelevant_pmids)} irrelevant.")
        for i, p in enumerate(irrelevant_pmids):
            print("*", i + 1, "*", p[0], "*", p[1])

    return correct_pmids, missed_pmids, irrelevant_pmids


# %%
logging.getLogger().setLevel(logging.CRITICAL)

# Open and load the .yaml file
with open("/home/azureuser/ev-agg-exp/lib/config/pubmed_library_config.yaml", "r") as file:
    yaml_data = yaml.safe_load(file)

# Get max_papers parameter
max_papers = yaml_data["library"]["max_papers"]  # TODO: use this parameter to limit the number of papers to fetch

# Get the query/ies
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

# Get the papers for each query, compare tool papers to the ground truth papers, and compare PubMed papers to the ground truth papers
for query in query_list_yaml:
    rare_disease_papers, count_r_d_papers, non_rare_disease_papers, other_papers = library.get_papers(query)
    term = query["gene_symbol"]

    if count_r_d_papers != 0:
        # Compare our tool papers (PMIDs) to the ground truth papers (PMIDs)
        correct_pmids_papers, missed_pmids_papers, irrelevant_pmids_papers = compare_to_truth_or_tool(
            term, rare_disease_papers, 0, ncbi_lookup
        )
        papers = rare_disease_papers.union(non_rare_disease_papers, other_papers)
    else:
        # No need to group rare_disease_papers as empty set.
        papers = non_rare_disease_papers.union(other_papers)

    # Compare PubMed papers to the ground truth papers
    pn_corr, pn_miss, pn_extra = compare_to_truth_or_tool(term, papers, 1, ncbi_lookup)

# %%
