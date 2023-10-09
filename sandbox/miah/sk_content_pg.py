"""sk_content_pg.py

Playground for experimentation on content extraction using SemanticKernel.
"""

# %% Imports.
import requests

# %% Constants.
PMID = 34758253
query_gene = "COQ2"
query_variant = None

# %% Handle Config.

# %% Load in, pre-process a paper.

# Get the paper from pubmed
# from Bio import Entrez
# Entrez.email = "miah@microsoft.com"

# handle = Entrez.efetch(db="pubmed", id=PMID, retmode="xml")
# records = Entrez.read(handle)
# handle.close()

pmid = PMID
"""Fetch a paper from PubMed Central using the BioC API."""
r = requests.get(f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmid}/ascii", timeout=10)
r.raise_for_status()
d = r.json()

# %%
if len(d["documents"]) != 1:
    raise ValueError(f"Expected a single document, got {len(d['documents'])}")

doc = d["documents"][0]

# %% exploring doc['passages']
# This is where the meat of the document is.

# each passage has an "infons" dict, which houses the relevant metadata
#   has "section_type"
#   has "type"
#
# each passage has a "text" field, which is the actual text of the passage

# set([p['infons']['section_type'] for p in doc['passages']])
# set([p['infons']['type'] for p in doc['passages']])

[len(p["text"]) for p in doc["passages"]]

# %% Use pubtator API for entity recognition.

pmcid = "PMC6912785"
format = "json"
r = requests.get(f"https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/bioc{format}?pmcids={pmcid}")
r.raise_for_status()
d = r.json()

# This gives us a fat pile of annotations of extracted entities, but it only gives you the official symbol for each gene,
# So we have to know that the query gene is the official one. This info can be pulled from NCBI's gene database.

# %% Ok, let's try to narrow down to passages that only contain the named entity.
for p in d["passages"]:
    for a in p["annotations"]:
        if a["text"] == "COQ2":
            print(f"COQ2 found in passage {p['infons']['section']}:")
            for l in a["locations"]:
                o = l["offset"] - p["offset"]
                print(f"  {p['text'][max(o-200, 0):(o+200)]}")

# %% Yeah, that works, but better yet, I think we can just use the variants (which have an RSID) and then go ask which
# of those RSIDs are on the query gene.

import json

# No, even better, the annotation includes the Gene ID.
from typing import Sequence


def _gene_symbol_for_id(ids: Sequence[str]) -> dict[str, str]:
    # TODO, wrap in Bio.Entrez library as they're better about rate limiting and such.
    url = f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha/gene/id/{','.join(ids)}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return {g["gene"]["gene_id"]: g["gene"]["symbol"] for g in r.json()["reports"]}


def _gene_id_for_symbol(symbols: Sequence[str]) -> dict[str, str]:
    # TODO, wrap in Bio.Entrez library as they're better about rate limiting and such.

    url = f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha/gene/symbol/{','.join(symbols)}/taxon/Human"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return {g["gene"]["symbol"]: g["gene"]["gene_id"] for g in r.json()["reports"]}


lookup = _gene_symbol_for_id(["4508", "12345"])
print(lookup)

foo = _gene_id_for_symbol(["COQ2", "KRAS"])
print(foo)

# TODO, random idea for later, point back to clinvar records for variants mentioned in papers

# %% Now that we can get gene ids, get the id for the query gene.

lookup = _gene_id_for_symbol([query_gene])
query_gene_id = int(lookup[query_gene])

# %% Then search all of the Mutation entities for the query gene id.

from typing import Any, List

variants_in_query_gene: List[dict[str, Any]] = []

for p in d["passages"]:
    for a in p["annotations"]:
        if a["infons"]["type"] == "Mutation":
            if "gene_id" in a["infons"] and int(a["infons"]["gene_id"]) == query_gene_id:
                variants_in_query_gene.append(a)


def _dedup_variants(variants_in: Sequence[dict[str, Any]]) -> Sequence[dict[str, Any]]:
    rsids = {v["infons"]["identifier"] for v in variants_in}
    variants = []
    for rsid in rsids:
        matches = [v for v in variants_in if v["infons"]["identifier"] == rsid]
        entity_ids = {v["id"] for v in matches}
        variant = matches[0]
        variant["entity_ids"] = entity_ids
        variants.append(variant)
    return variants


deduped = _dedup_variants(variants_in_query_gene)

# %% Now we've got a gene, and a list of variants within the document that are on that gene. Let's see what content
# we can extract about them.

# TODO: There are better ways to save this above
# get passages associated with each variant

for p in d["passages"]:
    in_passage = any(a["id"] in deduped[0]["entity_ids"] for a in p["annotations"])

    if in_passage:
        print(f"variant {[deduped[0]['text']]} found in passage:")
        print(f"  {p['text']}")

# %% Extract content.

# First we need to do entity extraction on the whole paper, looking for the gene of interest and any related variants.
# Once we have a list of variants to pursue, we can then start hunting down specific pieces of information about them.


# For each chunk, find chunks that are likely to contain content relevant to the query.

# For the potentially relevant chunks, extract candidate field values.

# We might have multiple candidates for a given field.  We need to resolve these.

# Now structure the result as a list of dicts, where the keys in each dict correspond to the fields, and each element
# in the list corresponds to a gene/variant pair.
