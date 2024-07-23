# Note, this notebook assumes interpreter state after running all the cells in study_set_gene_exploration.py

# # %%

# # https://ftp.ncbi.nlm.nih.gov/genomes/refseq/vertebrate_mammalian/Homo_sapiens/annotation_releases/current/GCF_000001405.40-RS_2023_10/GCF_000001405.40_GRCh38.p14_genomic.gff.gz
# # gunzip
# # cat GCF_000001405.40_GRCh38.p14_genomic.gff |  grep "MANE Select" | grep -P "BestRefSeq\\s+mRNA" > mane_select_transcripts_raw.txt
# # cat GCF_000001405.40_GRCh38.p14_genomic.gff |  grep "MANE Select" | grep -P "BestRefSeq\\s+CDS" > mane_select_proteins_raw.txt
#         # "DNAJC7": {  # No RefSeqGene accession, thus not in reference table. Using chromosomal genomic reference.
#         #     "GeneID": "7266",
#         #     "Symbol": "DNAJC7",
#         #     "RSG": "NC_000017.11",
#         #     "RNA": "NM_003315.4",
#         #     "Protein": "NP_003306.3",
#         # },

# refseqs = {}

# # Read mane_select_proteins_raw.txt as a dataframe with the columns
# with open("./mane_select_proteins_raw.txt", "r") as f:
#     for line in f.readlines():
#         tokens = line.split("\t")
#         assert len(tokens) == 9, "Wrong number of tokens in line"
#         attributes = {kv.split("=")[0]:kv.split("=")[1] for kv in tokens[8].split(";")}
#         assert "gene" in attributes, "gene not found in attributes"
#         assert "tag" in attributes and "Select" in attributes["tag"], "Not tagged as select"
#         assert "protein_id" in attributes, "protein_id not found in attributes"
#         assert "Dbxref" in attributes, "Dbxref not found in attributes"
#         xref = {kv.split(":")[0]:kv.split(":")[1] for kv in attributes["Dbxref"].split(",")}
#         assert "GeneID" in xref, "GeneID not found in DBxref"

#         if attributes["gene"] in refseqs:
#             if refseqs[attributes["gene"]]["MANE"]:
#                 continue
#             elif "MANE Select" in attributes["tag"]:
#                 print(f"Warning: {attributes['gene']} already has a non-MANE protein, replacing.")

#         refseqs[attributes["gene"]] = {
#             "Protein": attributes["protein_id"],
#             "RSG": tokens[0],
#             "Symbol": attributes["gene"],
#             "GeneID": xref["GeneID"],
#             "MANE": "MANE Select" in attributes["tag"]
#         }

# with open("./mane_select_transcripts_raw.txt", "r") as f:
#     for line in f.readlines():
#         tokens = line.split("\t")
#         assert len(tokens) == 9, "Wrong number of tokens in line"
#         attributes = {kv.split("=")[0]:kv.split("=")[1] for kv in tokens[8].split(";")}
#         assert "gene" in attributes, "gene not found in attributes"
#         assert "tag" in attributes and "Select" in attributes["tag"], "Not tagged as Select"
#         assert "transcript_id" in attributes, "transcript_id not found in attributes"

#         if attributes["gene"] not in refseqs:
#             print(f"Warning: {attributes['gene']} not found in proteins")
#             continue

#         if "RNA" in refseqs[attributes["gene"]]:
#             print(f"Warning: {attributes['gene']} already has an RNA")
#             continue

#         if "MANE Select" in attributes["tag"] and not refseqs[attributes["gene"]]["MANE"]:
#             print(f"Warning: {attributes['gene']} has a non-MANE protein, but a MANE RNA.")

#         refseqs[attributes["gene"]]["RNA"] = attributes["transcript_id"].strip()


# # %% Assess whether the study set genes are in refseqs

# for gene_struct in gene_config:
#     gene = gene_struct["gene_symbol"]
#     if gene not in refseqs:
#         print(f"Warning: {gene} not found in proteins")
#         continue


# # %%

# for k in refseqs.keys():
#     refseqs[k].pop("MANE")


# import json
# # write refseqs to file
# json.dump(refseqs, open("./.ref/refseqs.json", "w"), indent=4)
# #json.dumps(refseqs, "./.ref/refseqs.json")

# %% Play with the new refseq implementation

from lib.evagg.ref.refseq import RefSeqGeneLookupClient, RefSeqLookupClient
from lib.evagg.utils.web import RequestsWebContentClient

web_client = RequestsWebContentClient()
rsg_client = RefSeqGeneLookupClient(web_client)
rs_client = RefSeqLookupClient(web_client)

rs_client._lazy_init()


# %%

# %% Assess whether all the train/test genes are also in refseqs

import yaml

train_genes = yaml.safe_load(open("lib/config/queries/mgttrain_subset.yaml", "r"))

for gene_struct in train_genes:
    if not rsg_client.protein_accession_for_symbol(gene_struct["gene_symbol"]):
        print(f"Warning: {gene_struct['gene_symbol']} not found in rsg client (train)")
        continue

for gene_struct in train_genes:
    if not rs_client.protein_accession_for_symbol(gene_struct["gene_symbol"]):
        print(f"Warning: {gene_struct['gene_symbol']} not found in rs client (train)")
        continue

test_genes = yaml.safe_load(open("lib/config/queries/mgttest_subset.yaml", "r"))

for gene_struct in test_genes:
    if not rsg_client.protein_accession_for_symbol(gene_struct["gene_symbol"]):
        print(f"Warning: {gene_struct['gene_symbol']} not found in rsg client (test)")
        continue

for gene_struct in test_genes:
    if not rs_client.protein_accession_for_symbol(gene_struct["gene_symbol"]):
        print(f"Warning: {gene_struct['gene_symbol']} not found in rs client (test)")
        continue

study_set_genes = yaml.safe_load(open("lib/config/queries/study_set.yaml", "r"))

for gene_struct in study_set_genes:
    gene = gene_struct["gene_symbol"]
    if not rsg_client.protein_accession_for_symbol(gene):
        print(f"Warning: {gene} not found in rsg client (study set)")
        continue

for gene_struct in study_set_genes:
    gene = gene_struct["gene_symbol"]
    if not rs_client.protein_accession_for_symbol(gene):
        print(f"Warning: {gene} not found in rs client (study set)")
        continue

# %% Assess agreement for genes that exist in both the rsg and rs clients.

shared_keys = set(rsg_client._ref.keys()).intersection(set(rs_client._ref.keys()))

protein_accession_different = 0
transcript_accession_different = 0

genes_requiring_rerun = []

study_set_gene_symbols = [gene_struct["gene_symbol"] for gene_struct in study_set_genes]

for key in shared_keys:
    if (rsg := rsg_client.protein_accession_for_symbol(key)) != (rs := rs_client.protein_accession_for_symbol(key)):
        print(f"Warning: {key} has different protein accessions between rsg and rs clients. {rsg} != {rs}")
        protein_accession_different += 1
        if key in study_set_gene_symbols and key not in genes_requiring_rerun:
            genes_requiring_rerun.append(key)

    if (rsg := rsg_client.transcript_accession_for_symbol(key)) != (
        rs := rs_client.transcript_accession_for_symbol(key)
    ):
        print(f"Warning: {key} has different transcript accessions between rsg and rs clients. {rsg} != {rs}")
        transcript_accession_different += 1
        if key in study_set_gene_symbols and key not in genes_requiring_rerun:
            genes_requiring_rerun.append(key)

print(f"Protein accession differences: {protein_accession_different} (of {len(shared_keys)})")
print(f"Transcript accession differences: {transcript_accession_different} (of {len(shared_keys)})")


# %% Determine which study genes are in the rs client that aren't in the rsg client.
new_keys = set(rs_client._ref.keys()).difference(set(rsg_client._ref.keys()))

for key in new_keys:
    if key in study_set_gene_symbols:
        genes_requiring_rerun.append(key)

genes_requiring_rerun.sort()

# %% Get a list of genes we've already done content extraction for figure out how many don't actually need a rerun.

import os
import re

template = re.compile(r"extract_\d+_(.*)\.json")

elements = os.listdir(".out/run_evagg_pipeline_20240627_050854/results_cache/PromptBasedContentExtractor")

processed_genes = []

for elt in elements:
    result = template.findall(elt)
    if result and result[0] not in processed_genes:
        processed_genes.append(result[0])


genes_with_invalid_cache = list(set(processed_genes).intersection(set(genes_requiring_rerun)))
genes_with_invalid_cache.sort()

# %% Delete the cache items that need deleting.

for elt in elements:
    result = template.findall(elt)
    if result and result[0] in genes_with_invalid_cache:
        os.remove(f".out/run_evagg_pipeline_20240627_050854/results_cache/PromptBasedContentExtractor/{elt}")

# %%