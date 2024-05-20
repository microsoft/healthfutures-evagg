"""This script is an LLM prompt assistant for prompt engineering in paper finding. A copilot copilot."""

# Imports
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import warnings
from collections import defaultdict
from datetime import datetime
from functools import cache

from lib.di import DiContainer
from lib.evagg.llm import OpenAIClient
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.svc import get_dotenv_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)  # want to suppress pandas warning


def get_git_commit_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


@cache
def get_lookup_client() -> IPaperLookupClient:
    """Get the lookup client."""
    ncbi_lookup: IPaperLookupClient = DiContainer().create_instance({"di_factory": "lib/config/ncbi_lookup.yaml"}, {})
    return ncbi_lookup


def get_paper_abstracts(pmids: dict[str, str]) -> dict[str, str]:
    """Get the abstracts of papers given their PMIDs. TODO: Speed up via e.g. storing abstracts?"""
    # Get the lookup client
    client = get_lookup_client()
    abstracts = {}

    # Get the abstracts for each PMID
    for pmid in pmids:
        try:
            paper = client.fetch(pmid)
            abstracts[pmid] = paper.props.get("abstract", "Unknown") if paper else "Unknown"
        except Exception as e:
            ValueError(f"Error getting abstract for paper {pmid}: {e}")
    return abstracts


async def update_prompt(prompt_loc, misclass_papers) -> str:
    """Update the prompt given misclassified papers (from irrelevant papers for now)."""
    # Open the prompt and read its contents
    with open(prompt_loc, "r") as f:
        prompt = f.read()

    # Create an OpenAI client
    settings = get_dotenv_settings(filter_prefix="AZURE_OPENAI_")
    print(settings)
    client = OpenAIClient(settings)

    # Prompt the user to update the prompt
    response = await client.prompt_file(
        user_prompt_file=("lib/evagg/content/prompts/update_paper_finding_few_shot.txt"),
        params={"prompt": prompt, "dict_gene_titles_abstracts": misclass_papers},
    )
    return response


def create_misclassified_dict(filepath) -> dict[str, dict[str, list[str]]]:
    """Create a dictionary of misclassified papers from the benchmarking results."""
    # TODO: Update to include missed papers and not just irrelevant ones."""
    # Initialize the irrelevant papers dictionary
    irrelevant_papers = defaultdict(dict)

    # Initialize current gene
    current_gene = None

    # Initialize a flag to indicate whether we're in the irrelevant papers section
    in_irrelevant_section = False

    # Extract the irrelevant papers from the benchmarking results file
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("GENE:"):
                current_gene = line.split(":")[1].strip()
            elif re.match(r"Found E\.A\. \d+ irrelevant\.", line):
                in_irrelevant_section = True
            elif in_irrelevant_section and line.startswith("*"):
                _, _, pmid, title = line.split("*")
                irrelevant_papers[current_gene][pmid.strip()] = [title.strip(), "to_fill"]
            elif not line.startswith("*"):
                in_irrelevant_section = False

    # Get all PMIDs
    all_pmids = {pmid for pmids in irrelevant_papers.values() for pmid in pmids}

    # Get paper abstracts
    dict_pmids_abstracts = get_paper_abstracts(all_pmids)  # type: ignore

    # Replace the placeholder abstract with the actual abstract from dict_pmids_abstracts
    for gene_pmids in irrelevant_papers.values():
        for pmid, paper_info in gene_pmids.items():
            if pmid in dict_pmids_abstracts:
                paper_info[1] = dict_pmids_abstracts[pmid]

    return irrelevant_papers


# MAIN
# Run LLM prompt assistant to improve paper finding
main_prompt_file = "lib/evagg/content/prompts/paper_finding_few_shot.txt"

directory = (
    f".out/copilot_squared_paper_finding_results_{(datetime.today().strftime('%Y-%m-%d'))}_{get_git_commit_hash()}/"
)
os.makedirs(directory, exist_ok=True)
updated_prompt_file = f"{directory}paper_finding_few_shot_{datetime.today().strftime('%H:%M:%S')}.txt"


# Load in the misclassified papers (irrelevant only for now)
benchmark_results = directory + "/benchmarking_paper_finding_results_train.txt"

# Get the paper abstracts for the misclassified (irrelevant only for now) papers for a gene
misclass_papers = create_misclassified_dict(benchmark_results)

# Save last prompt to backup file
shutil.copyfile(main_prompt_file, updated_prompt_file)

# Update the paper_finding.txt prompt based on the misclassified (i.e. irrelevant) papers (misclass_papers)
# response = await update_prompt("lib/evagg/content/prompts/paper_finding_few_shot.txt", json.dumps(misclass_papers))
response = asyncio.run(
    update_prompt("lib/evagg/content/prompts/paper_finding_few_shot.txt", json.dumps(misclass_papers))
)

# Save the new prompt to the main prompt file
with open(main_prompt_file, "w") as f:
    f.write(response)
