"""This notebook is for playing around with the hgvs package."""

# Ideas to explore:
#  - gene proximity to variants
#  - prompt-based detection of gene mentions within a paper

# %% Imports.

from lib.evagg.lit.pubmed import PubtatorEntityAnnotator
from lib.evagg.types import Paper

from typing import Sequence, Dict, List
import requests

# %% Definitions.

def gene_symbols_for_id(gene_ids: Sequence[str]) -> Dict[str, List[str]]:
    seq_str = ",".join(gene_ids)

    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gene&id={seq_str}&format=json"
    response = requests.get(url)
    response.raise_for_status()

    response_dict = response.json()
    symbols: Dict[str, List[str]] = {}

    for gene_id in gene_ids:
        gene = response_dict["result"].get(gene_id, None)
        print(gene)
        if gene:
            symbols[gene_id] = [gene["name"]] + gene["otheraliases"].split(', ')

    return symbols

gene_symbols_for_id(["1738", "4508", "4540"])

# %% First, let's generate a candidate set of variants that we'll want to parse.

PMCIDS = ["PMC6912785", "PMC6175136"]

papers = [Paper(**{"id": pmcid, "is_pmc_oa": True, "pmcid": pmcid}) for pmcid in PMCIDS]
# Use pubtator to get variants for these PMIDs.

annotations = PubtatorEntityAnnotator().annotate(papers[0])

# %%

from lib.evagg.ref import NcbiGeneClient
from lib.evagg.web.entrez import BioEntrezClient, BioEntrezDotEnvConfig
gene_client = NcbiGeneClient(BioEntrezClient(BioEntrezDotEnvConfig()))

var_gene_ids = set()
for p in annotations['passages']:
    for a in p['annotations']:
        if a['infons']['type'] == 'Mutation':
            gene_id = a['infons'].get('gene_id', None)
            if gene_id:
                var_gene_ids.add(gene_id)

gene_symbols = set()
gene_identifiers = 
for p in annotations['passages']:
    for a in p['annotations']:
        if a['infons']['type'] == 'Gene':
            gene_symbols.add(a['text'])
            print(a)
            break

        

# %%

gene_client.gene_id_for_symbol('Pt17')

# %%

for p in annotations['passages']:
    for a in p['annotations']:
        if a['infons']['type'] == 'Mutation':
            if a['infons'].get('gene_id', 0) == 7634:
                print(a)
                
# %% Explore API

# create_variant()
#  - pass in Gene and c.dot
#            Gene and p.dot (but no transcript)
#            Gene, p.dot, and transcript
#            c.dot
#            chr, pos, ref, alt
#
# ensure that the variant is biologically plausible (ie., the reference allele is actually in the reference)
# %%

