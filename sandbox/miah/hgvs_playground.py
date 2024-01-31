"""This notebook is for playing around with the hgvs package."""

# Ideas to explore:
#  - gene proximity to variants
#  - prompt-based detection of gene mentions within a paper

# %% Imports.

import re
from typing import Any, Dict, List, Sequence

import requests

from lib.evagg.lit.pubmed import PubtatorEntityAnnotator
from lib.evagg.ref import NcbiGeneClient
from lib.evagg.types import Paper
from lib.evagg.web.entrez import BioEntrezClient, BioEntrezDotEnvConfig

# %% Definitions.


def gene_symbols_for_id(gene_ids: Sequence[str], max: int = -1) -> Dict[str, List[str]]:
    seq_str = ",".join(gene_ids)

    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gene&id={seq_str}&format=json"
    response = requests.get(url)
    response.raise_for_status()

    response_dict = response.json()
    symbols: Dict[str, List[str]] = {}

    for gene_id in gene_ids:
        gene = response_dict["result"].get(gene_id, None)
        if gene:
            symbol_list = [gene["name"]] + gene["otheraliases"].split(", ")
            if max >= 0:
                symbol_list = symbol_list[:max]
            symbols[gene_id] = symbol_list

    return symbols


# %% First, let's generate a candidate set of variants that we'll want to parse.

PMCIDS = ["PMC6912785", "PMC6175136"]

papers = [Paper(**{"id": pmcid, "is_pmc_oa": True, "pmcid": pmcid}) for pmcid in PMCIDS]
# Use pubtator to get variants for these PMIDs.

# %% Collect a bunch of variant names and corresponding genes from papers, it's ok of these are noisy, with
# incorrect gene associations (for example), since we're just going to use them as a smoke test
# for HGVS parsing.

finder = re.compile(r"NM_[0-9]+\.[0-9]+")

# gene_client = NcbiGeneClient(BioEntrezClient(BioEntrezDotEnvConfig()))


def get_nearest_transcript(tx_matches: Sequence[Any], loc: int, max_dist: int) -> str | None:
    closest_str = None
    closest_distance = max_dist + 1
    for match in tx_matches:
        distance = min(abs(x - loc) for x in match.span())
        if distance < closest_distance:
            closest_str = match.group()
            closest_distance = distance
    return closest_str


def mane_select_for_protein(gene_symbol: str) -> str:
    raise NotImplementedError()
    return ""


variant_tuples = []

for idx, paper in enumerate(papers):
    anno = PubtatorEntityAnnotator().annotate(paper)
    if not anno:
        continue
    print(f"Analyzing paper {idx} - {paper}")

    # Preprocess the annotations to get a gene symbol lookup
    gene_ids = set()
    for p in anno["passages"]:
        for a in p["annotations"]:
            if a["infons"]["type"] == "Gene":
                # sometimes they're semicolon delimited
                gene_ids.update(a["infons"]["identifier"].split(";"))

    gene_symbol_dict = gene_symbols_for_id(list(gene_ids), max=1)

    for p in anno["passages"]:
        tx_matches = [r for r in re.finditer(finder, p["text"])]
        for a in p["annotations"]:
            if a["infons"]["type"] == "Variant":
                v = {"mutation": a["text"]}

                gene_id_int = a["infons"].get("gene_id", None)
                if gene_id_int:
                    matching_symbols = gene_symbol_dict.get(str(gene_id_int), None)
                    if matching_symbols:
                        v["gene"] = matching_symbols[0]
                    else:
                        v["gene"] = None
                else:
                    v["gene"] = None

                if tx_matches:
                    v["transcript"] = get_nearest_transcript(tx_matches, a["locations"][0]["offset"] - p["offset"], 100)
                elif v["gene"]:
                    v["transcript"] = None
                else:
                    v["transcript"] = None

                variant_tuples.append(v)

# %% Let's see how well LitVar's autocomplete does.

from lib.evagg.ref.litvar import LitVarReference

result = LitVarReference().variant_autocomplete("c.737C>T", limit=5)

[r["gene"] for r in result]

# Anecdotally, litvar isn't great at this. I've cherry picked an example here, but it's not a *rare* case
# where litvar doesn't seem to pick up the correct gene for the variant. Interesting that pubtator seems to do better
# even though they're using litvar under the hood for autocomplete. I think the proximity matching in the source text
# is helping pubtator's performance.

# %% Try to parse these variants.

from hgvs.parser import Parser

parser = Parser()

v = parser.parse_hgvs_variant("NM_024996.7:c.689+908G>A")
v.validate()

# parse based on mutation and gene


# %%

import time
import xml.etree.ElementTree as ET


def check_mane_select(symbol: str) -> bool:
    url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=nuccore&retmode=json&term={symbol}[Gene] AND refseq_select[filter] AND "Homo sapiens"[Organism]'
    response = requests.get(url)
    response.raise_for_status()

    tx_entries = response.json()

    if tx_entries["esearchresult"]["count"] == "0":
        print(f"Warning ({symbol}): MANE select transcript entries")
        return False
    elif tx_entries["esearchresult"]["count"] != "1":
        print(f"Warning ({symbol}): multiple MANE select entries ({tx_entries['esearchresult']['idlist']})")
        return False

    tx_id = tx_entries["esearchresult"]["idlist"][0]

    # Get the whole nucleotide entry for this object ID
    url2 = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&rettype=native&retmode=xml&id={tx_id}"

    response2 = requests.get(url2)
    response2.raise_for_status()

    tx_result = response2.text

    time.sleep(1)

    return tx_result.find("<GB-block_keywords_E>MANE Select</GB-block_keywords_E>") >= 0


genes = set(d["gene"] for d in variant_tuples if d["gene"])

for gene in list(genes):
    if check_mane_select(gene):
        print(f"MANE select transcript found for {gene}")

# Here, on the original set of two papers, we see a couple interesting failure modes.
# There are MANE select transcripts for some genes, but not all of them.
# Warning (ERVK-1): MANE select transcript entries
# Warning (ND5): MANE select transcript entries
# Warning (ATP6): MANE select transcript entries
# Warning (TAS2R6P): MANE select transcript entries

# EVRK-1 is picked up as a synonym for CA1, which is actually mentioned in the second paper as a DOMAIN of interest in
# PRKCG. This is a misassignment of associated gene by pubtator. Potentially we need to consider a better method for
# gene association.


# symbol="SRSF1"
# url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=nuccore&retmode=json&term={symbol}[Gene] AND refseq_select[filter] AND "Homo sapiens"[Organism]'
# response = requests.get(url)
# response.raise_for_status()

# tx_entries = response.json()

# if tx_entries['esearchresult']['count'] == '0':
#     raise ValueError("No search results for gene symbol, do something different.")

# tx_id = tx_entries['esearchresult']['idlist'][0]

# url2 = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&rettype=native&retmode=xml&id={tx_id}'
# # The below URL will return just the refseq ID for the transcript (which should be the human mane select if one exists)
# # url2 = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&rettype=acc&retmode=text&id={tx_id}'
# response2 = requests.get(url2)
# response2.raise_for_status()

# tx_result = response2.text

# print(tx_result)

# # https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=nuccore&term=SRSF1[All%20Fields]%20AND%20refseq_select[filter]%20AND%20%22Homo%20sapiens%22[Organism]

# %%
