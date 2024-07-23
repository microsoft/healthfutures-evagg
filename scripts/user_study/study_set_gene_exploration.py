# %% Imports.

from time import sleep

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from lib.di import DiContainer

# %% Constants.

# This is a list of genes that were extracted from the specific cases used in the user study. The process for how
# these genes were selected is described in the user study protocol. 
GENE_PATH = "./scripts/user_study/study_set_genes.txt"

# These are genes from the above that we are opting to exclude from the user study.
TROUBLE_GENES = [
    "ABR",  # Many NCBI results, corresponds to a journal acronym
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
    "ACE2",  # Many NCBI results, covid gene
    "AF241726.2",
    "AL365214.2",
    "APOBEC3B-AS1",  # Many NCBI results
    "BCR",  # Many NCBI results, corresponds to a journal acronym
    "BX664727.3",
    "C21orf59-TCP10L",
    "CR381670.2",
    "CT47B1",
    "FAM160A2",
    "GOLGA8K",
    "GOLGA8S",
    "HLA-A",  # Many NCBI results
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

# %% Load in the study set genes.
# Study set genes are just a text file with one gene per line. Read this into a list.
with open(GENE_PATH, "r") as f:
    study_set_genes = [line.strip() for line in f.readlines()]

study_set_genes.sort()

# %% Make an NCBI client
ncbi_client = DiContainer().create_instance(spec={"di_factory": "lib/config/objects/ncbi.yaml"}, resources={})

# %% Run the NCBI fetch for each gene.

# Set a maximum number of papers to fetch for each gene, otherwise the Entrez client returns a relatively small set.
retmax = 1000

all_papers = {}

for index, gene in enumerate(study_set_genes):
    params = {"query": f"Gene {gene} pubmed pmc open access[filter]", "retmax": retmax}
    paper_ids = ncbi_client.search(**params)

    if len(paper_ids) == retmax:
        # If the number of papers returned is equal to retmax, there are likely more than retmax papers, which puts
        # this gene out of scope for the user study.
        print(f"Retmax reached for {gene}, skipping.")
        continue

    print(f"{index} of {len(study_set_genes)}: Fetching {len(paper_ids)} papers for {gene}.")


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

# %% Generate figures assessing various aspects of the study gene set and associated papers.

# Make a histogram that shows how many genes have [0, 10), [10, 20), [20, 30), etc. papers.
plt.figure(figsize=(12, 6))
sns.histplot(data=paper_df.groupby("gene").size(), bins=range(0, 100, 5), kde=False) # type: ignore

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
sns.histplot(data=paper_df.groupby("gene")["year"].min(), bins=range(1975, 2025, 1), kde=False) # type: ignore

plt.ylabel("Number of genes")
plt.xlabel("Year of first publication")


# %% Make a config yaml for the study genes.

# Note we're excluding genes that are in the TROUBLE_GENES list and genes that start with "HLA-"
# We're also setting a specific max_date here for repeatability. Finally, papers published before 2001 are generally
# always in PDF format and thus out of scope for the current pipeline implementation.
gene_config = [
    {"gene_symbol": gene, "min_date": "2001/01/01", "max_date": "2024/06/25", "retmax": 1000}
    for gene in study_set_genes
    if gene not in TROUBLE_GENES and gene.startswith("HLA-") == False
]

yaml.safe_dump(gene_config, open("lib/config/queries/study_set.yaml", "w"))
