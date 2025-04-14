#!/usr/bin/env python3
"""Script to download abstract or full text for papers related to genes in bench1.tsv.

This script:
1. Reads gene symbols and PMIDs from the bench1.tsv file
2. Searches PubMed for papers related to each gene symbol
3. Fetches papers directly by PMID from the bench1.tsv file
4. Downloads abstracts or full text when available
5. Saves the results as JSON files in .tmp/kg/[pmid].json
"""

# %% Imports
import asyncio
import json
import logging
import os
from typing import Dict, List, Set

import nest_asyncio

from lib.di import DiContainer
from lib.evagg.content.fulltext import get_fulltext
from lib.evagg.ref import NcbiLookupClient
from lib.evagg.types import Paper
from lib.evagg.utils import init_logger

nest_asyncio.apply()  # Apply the patch to allow nested event loops
# This is necessary for Jupyter notebooks or environments that already have an event loop running


# %% Constants
BENCH_FILE = "data/kg/bench1.tsv"
OUTPUT_DIR = ".tmp/kg"
MAX_GENES = None  # Set to an integer value to limit the number of genes processed
LOG_LEVEL = "INFO"

# %% Initialize logging
init_logger(level=LOG_LEVEL, to_file=True)
logger = logging.getLogger(__name__)


# %% Helper functions
def parse_bench_file(filepath: str) -> Dict[str, List[str]]:
    """Parse the bench1.tsv file to extract gene symbols and PMIDs.

    Args:
        filepath: Path to the bench1.tsv file

    Returns:
        Dictionary mapping gene symbols to lists of PMIDs
    """
    gene_to_pmids = {}
    all_pmids = set()

    with open(filepath, "r") as f:
        # Skip header line
        next(f)
        for line in f:
            if line.strip():
                gene, pmid = line.strip().split("\t")
                if gene not in gene_to_pmids:
                    gene_to_pmids[gene] = []
                gene_to_pmids[gene].append(pmid)
                all_pmids.add(pmid)

    logger.info(f"Found {len(gene_to_pmids)} genes and {len(all_pmids)} unique PMIDs in {filepath}")
    return gene_to_pmids


def ensure_tmp_dir(directory: str) -> None:
    """Ensure the temporary directory exists.

    Args:
        directory: Directory path to create
    """
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Created directory: {directory}")


def save_paper_json(paper: Paper, output_dir: str) -> None:
    """Save paper content as JSON.

    Args:
        paper: Paper object containing the paper data
        output_dir: Directory where to save the JSON file
    """
    pmid = paper.id
    pmid = pmid.lower().replace("pmid:", "")  # Normalize PMID format

    # Extract the data we need
    paper_data = {"pmid": pmid, "title": paper.props.get("title", ""), "text": paper.props.get("abstract", "")}

    # If full text is available, use it instead of abstract
    if paper.props.get("can_access", True) and len(paper.props.get("fulltext_xml", "")) > 0:
        full_text = get_fulltext(paper.props["fulltext_xml"], exclude=["AUTH_CONT", "ACK_FUND", "COMP_INT", "REF"])
        if full_text:
            paper_data["text"] = full_text
            print(f"Full text available for {pmid}, using it instead of abstract.")

    # Save to JSON file
    output_path = os.path.join(output_dir, f"{pmid}.json")
    with open(output_path, "w") as f:
        json.dump(paper_data, f, indent=2)

    logger.info(f"Saved paper {pmid} to {output_path}")


async def process_paper(pmid: str, ncbi_client: NcbiLookupClient, output_dir: str) -> None:
    """Process a single paper by PMID.

    Args:
        pmid: PMID of the paper to fetch
        ncbi_client: NCBI client to use for fetching
        output_dir: Directory where to save the result
    """
    try:
        # Fetch the paper with full text if available
        paper = ncbi_client.fetch(pmid, include_fulltext=True)
        if paper:
            save_paper_json(paper, output_dir)
        else:
            logger.warning(f"Failed to fetch paper {pmid}")
    except Exception as e:
        logger.error(f"Error processing paper {pmid}: {str(e)}")


async def process_gene(
    gene: str, pmids: List[str], ncbi_client: NcbiLookupClient, output_dir: str, processed_pmids: Set[str]
) -> None:
    """Process a gene by searching for papers and processing PMIDs.

    Args:
        gene: Gene symbol to search for
        pmids: List of PMIDs already known for this gene
        ncbi_client: NCBI client to use for fetching
        output_dir: Directory where to save results
        processed_pmids: Set of PMIDs that have already been processed
    """
    try:
        # Process PMIDs specifically listed for this gene
        for pmid in pmids:
            if pmid not in processed_pmids:
                logger.info(f"Processing truthset PMID {pmid} for gene {gene}")
                await process_paper(pmid, ncbi_client, output_dir)
                processed_pmids.add(pmid)
            else:
                logger.info(f"truthset PMID {pmid} already processed, skipping.")

        # Search for additional papers on this gene
        params = {"query": gene, "retmax": 1000}
        search_results = ncbi_client.search(**params)

        logger.info(f"Found {len(search_results)} papers for gene {gene}")
        if len(search_results) == params["retmax"]:
            logger.warning(
                f"Reached the maximum number of papers ({params['retmax']}) for gene {gene}. This may cause "
                "an incomplete result, with not all papers matching the gene being processed."
            )

        # Process each search result
        for pmid in search_results:
            if pmid not in processed_pmids:
                logger.info(f"Processing search PMID {pmid} for gene {gene}")
                await process_paper(pmid, ncbi_client, output_dir)
                processed_pmids.add(pmid)
            else:
                logger.info(f"Search PMID {pmid} already processed, skipping.")

    except Exception as e:
        logger.error(f"Error processing gene {gene}: {str(e)}")


# %% Initialize NCBI client using DI
ncbi_client = DiContainer().create_instance({"di_factory": "lib/config/objects/ncbi_cache.yaml"}, {})

# %% Parse the bench file
gene_to_pmids = parse_bench_file(BENCH_FILE)

# %% Ensure output directory exists
ensure_tmp_dir(OUTPUT_DIR)


# %% Process genes and PMIDs
async def process_all_genes():
    # Track processed PMIDs to avoid duplicates
    processed_pmids: Set[str] = set()

    # Process each gene and its PMIDs
    gene_list = list(gene_to_pmids.keys())
    if MAX_GENES:
        gene_list = gene_list[:MAX_GENES]

    for gene in gene_list:
        logger.info(f"Processing gene: {gene}")
        await process_gene(gene, gene_to_pmids[gene], ncbi_client, OUTPUT_DIR, processed_pmids)

    logger.info(f"Processed {len(processed_pmids)} unique PMIDs for {len(gene_list)} genes")


# %% Run the processing

if __name__ == "__main__":
    asyncio.run(process_all_genes())

# %% Looks like paper 33471991 can't use the caching client, let's do it manually here with without the _cache client.

ncbi_client_no_cache = DiContainer().create_instance({"di_factory": "lib/config/objects/ncbi.yaml"}, {})
pmid = "33471991"
paper = ncbi_client_no_cache.fetch(pmid, include_fulltext=True)
save_paper_json(paper, OUTPUT_DIR)
