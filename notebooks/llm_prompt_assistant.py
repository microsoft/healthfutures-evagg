"""This script is used to implement an LLM assistant to aid in prompt engineering for paper finding.
"""

# %% Imports.
import json
import logging
import os
import pickle
import warnings
from datetime import datetime
from functools import cache
from typing import Dict, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from lib.di import DiContainer
from lib.evagg.library import RareDiseaseFileLibrary
from lib.evagg.llm import OpenAIClient
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import Paper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)  # want to suppress pandas warning

# %% Read in the truth and output tables.


@cache
def get_lookup_client() -> IPaperLookupClient:
    ncbi_lookup: IPaperLookupClient = DiContainer().create_instance({"di_factory": "lib/config/ncbi_lookup.yaml"}, {})
    return ncbi_lookup


def get_paper_titles_abstracts(pmids: Set[str]):
    # This increases execution time a fair amount, we could alternatively store the titles in the MGT and the pipeline
    # output to speed things up if we wanted.
    client = get_lookup_client()
    titles = {}
    abstracts = {}
    for pmid in pmids:
        try:
            paper = client.fetch(pmid)
            titles[pmid] = paper.props.get("title", "Unknown") if paper else "Unknown"
            abstracts[pmid] = paper.props.get("abstract", "Unknown") if paper else "Unknown"
        except Exception as e:
            print(f"Error getting title for paper {pmid}: {e}")
            titles[pmid] = "Unknown"
    return titles, abstracts


# def get_llm_category(pmid, title, abstract) -> str:
#     """Categorize papers based on LLM prompts."""
#     response = llm_client.prompt_file(
#         user_prompt_file=os.path.join(os.path.dirname(__file__), "content", "prompts", "paper_finding.txt"),
#         system_prompt="Extract field",
#         params={
#             "abstract": abstract,
#             "title": title,
#         },
#         prompt_settings={"prompt_tag": "paper_category"},
#     )

#     try:
#         result = json.loads(response)["paper_category"]
#     except Exception:
#         result = response

#     if result in CATEGORIES:
#         return result

#     logger.warning(f"LLM failed to return a valid categorization response for {pmid}: {response}")
#     return "other"


# %%
with open("/home/azureuser/ev-agg-exp/.out/paper_finding_results_2024-04-11/irrelevant_pmids_all_genes.pkl", "rb") as f:
    irrelevant_pmids_all_genes = pickle.load(f)

print("irrelevant_pmids_all_genes", irrelevant_pmids_all_genes)

titles, abstracts = get_paper_titles_abstracts(set(irrelevant_pmids_all_genes["EXOC2"]))

print("titles", titles)
print("abstracts", abstracts)

# %%

library: RareDiseaseFileLibrary = DiContainer().create_instance({"di_factory": "lib/config/library.yaml"}, {})
ncbi_lookup: IPaperLookupClient = DiContainer().create_instance({"di_factory": "lib/config/ncbi_lookup.yaml"}, {})

llm_cat = get_llm_category(paper)
print(llm_cat)

# %%
for gene, pmids in irrelevant_pmids_all_genes.items():
    print("gene", gene)
    print("pmids", pmids)
    dict_pmids_titles, dict_pmids_abstracts = get_paper_titles_abstracts(pmids)
    for pmid in dict_pmids_titles.keys():
        print(dict_pmids_titles[pmid])
        print(dict_pmids_abstracts[pmid])
        params = {"abstract": dict_pmids_titles[pmid] or "Unknown", "title": dict_pmids_abstracts[pmid] or "Unknown"}
        client = OpenAIClient(
            {"deployment": "gpt-8", "endpoint": "https://ai", "api_key": "test", "api_version": "test"}
        )
        response = client.prompt_file(
            user_prompt_file=("lib/evagg/content/prompts/paper_finding.txt"),
            system_prompt="Extract field",
            params={
                "title": dict_pmids_titles[pmid],
                "abstract": dict_pmids_abstracts[pmid],
            },  # type: ignore
            prompt_settings={"prompt_tag": "paper_category"},
        )
        # try:
        #     result = json.loads(response)["paper_category"]
        # except Exception:
        #     result = "failed"  # TODO: how to handle this?
        # Categorize the paper based on LLM result
        print("result", response)
    # if result in paper_categories:
    #     paper_categories[result].add(paper)
    # elif result == "failed":
    #     logger.warning(f"Failed to categorize paper: {paper.id}")
    # else:
    #     raise ValueError(f"Unexpected result: {result}")

    # return paper_categories["rare disease"], paper_categories["non-rare disease"], paper_categories["other"]

# %%
# def main(args):

#     # Open and load the .yaml file
#     with open(args.library_config, "r") as file:
#         yaml_data = yaml.safe_load(file)

#     # Read the intermediate manual ground truth (MGT) data file from the TSV file
#     mgt_gene_pmids_dict = read_mgt_split_tsv(args.mgt_train_test_path)

#     # Get the query/ies from .yaml file
#     if ".yaml" in str(yaml_data["queries"]):  # leading to query .yaml
#         with open(yaml_data["queries"]["di_factory"], "r") as file:
#             yaml_data = yaml.safe_load(file)
#             query_list_yaml = read_queries(yaml_data)
#     elif yaml_data["queries"] is not None:
#         query_list_yaml = read_queries(yaml_data["queries"])
#     else:
#         raise ValueError("No queries found in the .yaml file.")

#     # Create the library, the PubMed lookup clients, and the LLM client
#     library: RareDiseaseFileLibrary = DiContainer().create_instance({"di_factory": "lib/config/library.yaml"}, {})
#     ncbi_lookup: IPaperLookupClient = DiContainer().create_instance({"di_factory": "lib/config/ncbi_lookup.yaml"}, {})

#     # Initialize the dictionary
#     benchmarking_results = {}

#     # Average precision and recall for all genes
#     avg_precision = []
#     avg_recall = []

#     # For each query, get papers, compare ev. agg. papers to MGT data papers,
#     # compare PubMed papers to MGT data papers. Write results to benchmarking against MGT file.
#     os.makedirs(args.outdir, exist_ok=True)
#     with open(os.path.join(args.outdir, "benchmarking_paper_finding_results_train.txt"), "w") as f:
#         for query in query_list_yaml:

#             # Get the gene name from the query
#             term = query["gene_symbol"]
#             f.write(f"\nGENE: {term}\n")
#             print("Finding papers for: ", query["gene_symbol"], "...")

#             # Initialize the list for this gene
#             benchmarking_results[term] = [0] * 10

#             # Get  papers from library for the query (i.e. gene/term), TODO: better to union 3 sets or keep 'papers'?
#             (rare_disease_papers, non_rare_disease_papers, other_papers, papers,
#                 discordant_human_in_loop, counts_discordant_hil) = library.get_all_papers(query)

#             # Check if =<3 papers categories are empty, and report the number of papers in each category
#             if not rare_disease_papers:
#                 f.write(f"Rare Disease Papers: {0}\n")
#                 benchmarking_results[term][0] = 0
#             else:
#                 f.write(f"Rare Disease Papers: {len(rare_disease_papers)}\n")
#                 benchmarking_results[term][0] = len(rare_disease_papers)
#             if not non_rare_disease_papers:
#                 f.write(f"Non-Rare Disease Papers: {0}\n")
#                 benchmarking_results[term][1] = 0
#             else:
#                 f.write(f"Non-Rare Disease Papers: {len(non_rare_disease_papers)}\n")
#                 benchmarking_results[term][1] = len(non_rare_disease_papers)
#             if not other_papers:
#                 f.write(f"Other Papers: {0}\n")
#                 benchmarking_results[term][2] = 0
#             else:
#                 f.write(f"Other Papers: {len(other_papers)}\n")
#                 benchmarking_results[term][2] = len(other_papers)
#             if not discordant_human_in_loop:
#                 f.write(f"Discordant Papers: {0}\n")
#                 benchmarking_results[term][3] = 0
#             else:
#                 f.write(f"Discordant Papers: {len(discordant_human_in_loop)}\n")
#                 benchmarking_results[term][3] = len(discordant_human_in_loop)

#                 for i, p in enumerate(discordant_human_in_loop):
#                     f.write(f"* {i + 1} * {p.props.get("pmid", "Unknown")} * {p.props.get("title", "Unknown")}\n")
#                     f.write(f"* {i + 1}-counts * {counts_discordant_hil[i]}\n")

#             # If ev. agg. found rare disease papers, compare ev. agg. papers (PMIDs) to MGT data papers (PMIDs)
#             print("Comparing Evidence Aggregator results to manual ground truth data for:", query["gene_symbol"], "...")
#             if rare_disease_papers != Set[Paper]:
#                 correct_pmids, missed_pmids, irrelevant_pmids = compare_to_truth_or_tool(
#                     term, rare_disease_papers, ncbi_lookup, mgt_gene_pmids_dict
#                 )

#                 # Report comparison between ev.agg. and MGT data
#                 f.write("\nOf Ev. Agg.'s rare disease papers...\n")
#                 f.write(f"E.A. # Correct Papers: {len(correct_pmids)}\n")
#                 f.write(f"E.A. # Missed Papers: {len(missed_pmids)}\n")
#                 f.write(f"E.A. # Irrelevant Papers: {len(irrelevant_pmids)}\n")

#                 # Calculate precision and recall
#                 if len(correct_pmids):
#                     precision, recall, f1_score = calculate_metrics(correct_pmids, missed_pmids, irrelevant_pmids)

#                     f.write(f"\nPrecision: {precision}\n")
#                     f.write(f"Recall: {recall}\n")
#                     f.write(f"F1 Score: {f1_score}\n")
#                     avg_precision.append(precision)
#                     avg_recall.append(recall)
#                 else:
#                     f.write("\nNo true positives. Precision and recall are undefined.\n")

#                 # Update the metrics in the list for this gene
#                 benchmarking_results[term][4] = len(correct_pmids)
#                 benchmarking_results[term][5] = len(missed_pmids)
#                 benchmarking_results[term][6] = len(irrelevant_pmids)

#                 f.write(f"\nFound E.A. {len(correct_pmids)} correct.\n")
#                 for i, p in enumerate(correct_pmids):
#                     f.write(f"* {i + 1} * {p[0]} * {p[1]}\n")  # PMID and title output

#                 f.write(f"\nFound E.A. {len(missed_pmids)} missed.\n")
#                 for i, p in enumerate(missed_pmids):
#                     f.write(f"* {i + 1} * {p[0]} * {p[1]}\n")  # PMID and title output

#                 f.write(f"\nFound E.A. {len(irrelevant_pmids)} irrelevant.\n")
#                 for i, p in enumerate(irrelevant_pmids):
#                     f.write(f"* {i + 1} * {p[0]} * {p[1]}\n")  # PMID and title output

#             else:
#                 f.write("\nOf Ev. Agg.'s rare disease papers...\n")
#                 f.write(f"E.A. # Correct Papers: {0}\n")
#                 f.write(f"E.A. # Missed Papers: {0}\n")
#                 f.write(f"E.A. # Irrelevant Papers: {0}\n")

#             # Compare PubMed papers to  MGT data papers
#             print("Comparing PubMed results to manual ground truth data for: ", query["gene_symbol"], "...")
#             p_corr, p_miss, p_irr = compare_to_truth_or_tool(term, papers, ncbi_lookup, mgt_gene_pmids_dict)
#             f.write("\nOf PubMed papers...\n")
#             f.write(f"Pubmed # Correct Papers: {len(p_corr)}\n")
#             f.write(f"Pubmed # Missed Papers: {len(p_miss)}\n")
#             f.write(f"Pubmed # Irrelevant Papers: {len(p_irr)}\n")

#             # Update the counts in the list for this gene
#             benchmarking_results[term][7] = len(p_corr)
#             benchmarking_results[term][8] = len(p_miss)
#             benchmarking_results[term][9] = len(p_irr)

#             f.write(f"\nFound Pubmed {len(p_corr)} correct.\n")
#             for i, p in enumerate(p_corr):
#                 f.write(f"* {i + 1} * {p[0]} * {p[1]}\n")  # PMID and title output

#             f.write(f"\nFound Pubmed {len(p_miss)} missed.\n")
#             for i, p in enumerate(p_miss):
#                 f.write(f"* {i + 1} * {p[0]} * {p[1]}\n")  # PMID and title output

#             f.write(f"\nFound Pubmed {len(p_irr)} irrelevant.\n")
#             for i, p in enumerate(p_irr):
#                 f.write(f"* {i + 1} * {p[0]} * {p[1]}\n")  # PMID and title output

#         # Calculate average precision and recall
#         if len(avg_precision) != 0:
#             avg_precision = sum(avg_precision) / len(avg_precision)
#             avg_recall = sum(avg_recall) / len(avg_recall)

#             # Write average precision and recall to the file
#             f.write(f"\nAverage Precision: {avg_precision}\n")
#             f.write(f"Average Recall: {avg_recall}\n")
#         else:
#             f.write("\nNo true positives. Precision and recall are undefined.\n")

#     # Plot benchmarking results
#     results_to_plot = plot_benchmarking_results(benchmarking_results)

#     # Plot filtering results
#     plot_filtered_categories(results_to_plot)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Evidence Aggregator Paper Finding Benchmarks")
#     parser.add_argument(
#         "-l",
#         "--library-config",
#         nargs="?",
#         default="lib/config/pubmed_library_config.yaml",
#         type=str,
#         help="Default is lib/config/pubmed_library_config.yaml",
#     )
#     parser.add_argument(
#         "-m",
#         "--mgt-train-test-path",
#         nargs="?",
#         default="data/v1/papers_train_v1.tsv",
#         type=str,
#         help="Default is data/v1/papers_train_v1.tsv",
#     )
#     parser.add_argument(
#         "--outdir",
#         default=f".out/paper_finding_results_{(datetime.today().strftime('%Y-%m-%d'))}",
#         type=str,
#         help=(
#             "Results output directory. Default is "
#             f".out/paper_finding_results_{(datetime.today().strftime('%Y-%m-%d'))}/"
#         ),
#     )
#     args = parser.parse_args()

#     main(args)

# %%
