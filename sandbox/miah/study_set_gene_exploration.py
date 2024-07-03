# %% Imports.

from time import sleep

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from lib.di import DiContainer

# %% Constants.

GENE_PATH = "./sandbox/miah/study_set_genes.txt"

TROUBLE_GENES = [
    "ABR",  # tons of papers, corresponds to a journal acronym
    "AC009061.2",
    "AC011322.1",
    "AC026740.3",
    "AC084121.13",
    "AC090983.2",
    "AC099489.1",
    "AC119396.1",
    "AC126323.1",
    "AC233976.1",
    "AC245452.1",
    "ACE2",  # Zillions of papers, covid gene
    "AF241726.2",
    "AL365214.2",
    "APOBEC3B-AS1",  # tons of papers
    "BCR",  # tons of papers, corresponds to a journal acronym
    "BX664727.3",
    "C21orf59-TCP10L",
    "CR381670.2",
    "CT47B1",
    "FAM160A2",
    "GOLGA8K",
    "GOLGA8S",
    "HLA-A",  # Overly common, don't process?
    "HLA-DRB1",
    "KRTAP10-7",
    "KRTAP2-2",
    "KRTAP4-8",
    "LINC01967",
    "MIR4527HG",
    "PARP1",
    "PML",
    "PNMA6F",
    "PRAMEF14",
    "SDR42E2",
    "TAF11L11",
    "TAF11L12",
    "UBALD2",
    "CLUHP8",
    "FRG1GP",
    "ST20-MTHFS",
    "TLR7",
]

# non-coding genes, we'll leave out
#

# Other interesting genes
# BCL6 - 618 open access papers, this is a known oncogene and well studied.
# FASN - 847 papers. Well studied metabolic gene. No gene-disease curations in gencc
# HLA-* - all HLA prefixed genes?

# %% Load in the study set genes.
# Study set genes are just a text file with one gene per line. Read this into a list.
with open(GENE_PATH, "r") as f:
    study_set_genes = [line.strip() for line in f.readlines()]

study_set_genes.sort()

# %% Make an NCBI client

ncbi_client = DiContainer().create_instance(spec={"di_factory": "lib/config/objects/ncbi.yaml"}, resources={})

# %% Run the NCBI fetch for each gene.

# params = {"query": query["gene_symbol"]}
# # Rationalize the optional parameters.
# if ("max_date" in query or "date_type" in query) and "min_date" not in query:
#     raise ValueError("A min_date is required when max_date or date_type is provided.")
# if "min_date" in query:
#     params["min_date"] = query["min_date"]
#     params["date_type"] = query.get("date_type", "pdat")
# if "max_date" in query:
#     params["max_date"] = query["max_date"]
# if "retmax" in query:
#     params["retmax"] = query["retmax"]

retmax = 1000

all_papers = {}

for index, gene in enumerate(study_set_genes):
    params = {"query": f"Gene {gene} pubmed pmc open access[filter]", "retmax": retmax}
    paper_ids = ncbi_client.search(**params)
    #    sleep(0.1)

    if len(paper_ids) == retmax:
        print(f"Retmax reached for {gene}, skipping.")
        continue

    print(f"{index} of {len(study_set_genes)}: Fetching {len(paper_ids)} papers for {gene}.")
    # for paper_id in paper_ids:
    #     try:
    #         all_papers[paper_id] = (gene, ncbi_client.fetch(paper_id, include_fulltext=True))
    #         sleep(0.1)
    #     except Exception as e:
    #         print(f"Error fetching paper {paper_id}: {e}")


# %% Convert all_papers to a dataframe with the following columns
# - gene
# - paper_id
# - title
# - can_access
# - has_fulltext
# - year


def year_from_citation(citation: str):
    return citation.split("(")[1].split(")")[0]


paper_data = []
for gene, paper in all_papers.values():
    paper_data.append(
        {
            "gene": gene,
            "paper_id": paper.id,
            "title": paper.props["title"],
            "can_access": paper.props["can_access"],
            "has_fulltext": paper.props["fulltext_xml"] != "",
            "year": year_from_citation(paper.props["citation"]),
        }
    )

paper_df = pd.DataFrame(paper_data)

# %% Get a list of query genes that aren't included in paper_df.
# This is a list of genes that we couldn't find any papers for.
missing_genes = [gene for gene in study_set_genes if gene not in paper_df["gene"].unique()]

print(f"Zero papers or skipped {len(missing_genes)} genes.")
for g in missing_genes:
    print("  ", g)

# %%

# Make a histogram that shows how many genes have [0, 10), [10, 20), [20, 30), etc. papers.
plt.figure(figsize=(12, 6))
sns.histplot(data=paper_df.groupby("gene").size(), bins=range(0, 100, 5), kde=False)

# x-axis label should be "Number of papers"
plt.xlabel("Number of OA/non-ND papers")
plt.ylabel("Number of genes")

# %% Make a plot of how many papers we have for each year, making sure the x axis labels are sorted
# in ascending order.
paper_df["year"] = paper_df["year"].astype(int)
paper_df = paper_df.sort_values("year")

plt.figure(figsize=(12, 6))
sns.countplot(data=paper_df, x="year")
plt.xticks(rotation=45)

# %% Make the same plot, but only for papers that have full text available.
plt.figure(figsize=(12, 6))
sns.countplot(data=paper_df[paper_df["has_fulltext"]], x="year")
plt.xticks(rotation=45)

# %% Plot a histogram showing the distribution of the year of the first publication for each gene.
# This is a histogram of the minimum year for each gene.
plt.figure(figsize=(12, 6))
sns.histplot(data=paper_df.groupby("gene")["year"].min(), bins=range(1975, 2025, 1), kde=False)

plt.ylabel("Number of genes")
plt.xlabel("Year of first publication")


# %% Make a config yaml for the study genes.

gene_config = [
    {"gene_symbol": gene, "min_date": "2001/01/01", "max_date": "2024/06/25", "retmax": 1000}
    for gene in study_set_genes
    if gene not in TROUBLE_GENES and gene.startswith("HLA-") == False
]

yaml.safe_dump(gene_config, open("lib/config/queries/study_set.yaml", "w"))


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
