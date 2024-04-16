"""This script is used to implement an LLM assistant to aid in prompt engineering for paper finding.
"""

# %% Imports.
import json
import logging
import os
import pickle
import re
import shutil
import warnings
from datetime import datetime
from functools import cache
from typing import Dict, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from lib.di import DiContainer
from lib.evagg.library import RareDiseaseFileLibrary
from lib.evagg.llm import OpenAIClient
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import Paper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)  # want to suppress pandas warning


@cache
def get_lookup_client() -> IPaperLookupClient:
    ncbi_lookup: IPaperLookupClient = DiContainer().create_instance({"di_factory": "lib/config/ncbi_lookup.yaml"}, {})
    return ncbi_lookup


def get_paper_titles_abstracts(pmids: dict[str, str]):
    # This increases execution time a fair amount, we could alternatively store the titles in the MGT and the pipeline
    # output to speed things up if we wanted.
    client = get_lookup_client()
    abstracts = {}
    for pmid in pmids:
        try:
            paper = client.fetch(pmid)
            # titles[pmid] = paper.props.get("title", "Unknown") if paper else "Unknown"
            abstracts[pmid] = paper.props.get("abstract", "Unknown") if paper else "Unknown"
        except Exception as e:
            print(f"Error getting title for paper {pmid}: {e}")
            # titles[pmid] = "Unknown"
    return abstracts


def update_prompt(prompt_loc, misclass_papers):
    """Categorize papers based on LLM prompts."""

    # Open the file and read its contents
    with open(prompt_loc, "r") as f:
        prompt = f.read()

    # Load environment variables from .env file
    load_dotenv()

    # Get the values of the environment variables
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    print("deployment", deployment, "endpoint", endpoint, "api_key", api_key, "api_version", api_version)

    client = OpenAIClient(
        {"deployment": deployment, "endpoint": endpoint, "api_key": api_key, "api_version": api_version}
    )
    response = client.prompt_file(
        user_prompt_file=("lib/evagg/content/prompts/update_paper_finding_prompt.txt"),
        params={"prompt": prompt, "dict_gene_titles_abstracts": misclass_papers},
    )
    return response


def get_llm_category(pmid, title, abstract) -> str:
    """Categorize papers based on LLM prompts."""
    categories = ["rare disease", "non-rare disease", "other"]

    # Load environment variables from .env file
    load_dotenv()

    # Get the values of the environment variables
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    print("deployment", deployment, "endpoint", endpoint, "api_key", api_key, "api_version", api_version)

    client = OpenAIClient(
        {"deployment": deployment, "endpoint": endpoint, "api_key": api_key, "api_version": api_version}
    )
    response = client.prompt_file(
        user_prompt_file=("lib/evagg/content/prompts/paper_finding.txt"),
        system_prompt="Extract field",
        params={
            "title": title,
            "abstract": abstract,
        },  # type: ignore
        prompt_settings={"prompt_tag": "paper_category"},
    )
    try:
        result = json.loads(response)["paper_category"]
    except Exception:
        result = response

    if result in categories:
        return result

    logger.warning(f"LLM failed to return a valid categorization response for {pmid}: {response}")
    return "failed"


def get_irrelevant_pmids():
    with open(
        "/home/azureuser/ev-agg-exp/.out/paper_finding_results_2024-04-11/irrelevant_pmids_all_genes.pkl", "rb"
    ) as f:
        irrelevant_pmids_all_genes = pickle.load(f)

    print("irrelevant_pmids_all_genes", irrelevant_pmids_all_genes)

    library: RareDiseaseFileLibrary = DiContainer().create_instance({"di_factory": "lib/config/library.yaml"}, {})
    ncbi_lookup: IPaperLookupClient = DiContainer().create_instance({"di_factory": "lib/config/ncbi_lookup.yaml"}, {})

    for gene, pmids in irrelevant_pmids_all_genes.items():
        print("gene", gene)
        print("pmids", pmids)
        dict_pmids_titles, dict_pmids_abstracts = get_paper_titles_abstracts(pmids)

        for pmid in dict_pmids_titles.keys():
            print(dict_pmids_titles[pmid])
            print(dict_pmids_abstracts[pmid])
            params = {
                "abstract": dict_pmids_titles[pmid] or "Unknown",
                "title": dict_pmids_abstracts[pmid] or "Unknown",
            }

            updated_prompt = update_prompt(pmid, dict_pmids_titles[pmid], dict_pmids_abstracts[pmid])
            print("HEY!", updated_prompt)


def create_misclassified_dict(filepath):
    # Open the file and read its contents
    with open(
        filepath,
        "r",
    ) as f:
        lines = f.read().splitlines()

    # Initialize the dictionary
    irrelevant_papers = {}

    # Initialize current gene
    current_gene = None

    # Initialize a flag to indicate whether we're in the irrelevant papers section
    in_irrelevant_section = False

    # Extract the irrelevant papers
    for line in lines:
        if line.startswith("GENE:"):
            current_gene = line.split(":")[1].strip()
        elif re.match(r"Found E\.A\. \d+ irrelevant\.", line):
            in_irrelevant_section = True
        elif in_irrelevant_section and line.startswith("*"):
            parts = line.split("*")
            pmid = parts[2].strip()
            title = parts[3].strip()
            abstract = "to_fill"  # placeholder for abstract, replace with actual extraction logic if available
            if current_gene not in irrelevant_papers:
                irrelevant_papers[current_gene] = {}
            irrelevant_papers[current_gene][pmid] = [title, abstract]
        elif not line.startswith("*"):
            in_irrelevant_section = False

    # Initialize an empty set
    all_pmids = set()

    # Loop through all genes in the dictionary
    for gene in irrelevant_papers:
        # Add the PMIDs of the current gene to the set
        all_pmids.update(irrelevant_papers[gene].keys())

    # Now you can pass all_pmids to the function
    dict_pmids_abstracts = get_paper_titles_abstracts(all_pmids)

    # Loop through all genes in the dictionary
    for gene in irrelevant_papers:
        # Loop through all PMIDs of the current gene
        for pmid in irrelevant_papers[gene]:
            # Replace the placeholder abstract with the actual abstract from dict_pmids_abstracts
            if pmid in dict_pmids_abstracts:
                irrelevant_papers[gene][pmid][1] = dict_pmids_abstracts[pmid]

    return irrelevant_papers


# %% Run LLM prompt assistant to improve paper finding.

# Load in the misclassified papers.
filepath = (
    "/home/azureuser/ev-agg-exp/.out/paper_finding_results_2024-04-11/benchmarking_paper_finding_results_train.txt"
)

# Get the paper titles and abstracts for the irrelevant papers for a gene.
misclass_papers = create_misclassified_dict(filepath)

# Save last prompt to backup file
main_prompt_file = "lib/evagg/content/prompts/paper_finding.txt"
directory = f".out/paper_finding_results_{datetime.today().strftime('%Y-%m-%d')}/"
updated_prompt_file = f"{directory}paper_finding_prompt_{datetime.today().strftime('%H:%M:%S')}.txt"
os.makedirs(directory, exist_ok=True)

shutil.copyfile(main_prompt_file, updated_prompt_file)

# Update the paper_finding.txt prompt based on the misclassified papers (misclass_papers)
response = update_prompt("lib/evagg/content/prompts/paper_finding.txt", json.dumps(misclass_papers))

# Save the new prompt to the main prompt file
with open(main_prompt_file, "w") as f:
    f.write(response)
