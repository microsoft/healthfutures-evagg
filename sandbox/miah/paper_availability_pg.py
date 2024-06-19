"""Notebook to explore whether/which all papers in the truth set are available to us via pubmed search."""

# %% Imports.

import pandas as pd
import yaml

from lib.di import DiContainer

# %% Constants.

PAPERS_TSVS = ["data/v1/papers_test_v1.tsv", "data/v1/papers_train_v1.tsv"]

# %% Load in the papers.

truth_papers = pd.concat([pd.read_csv(papers_tsv, sep="\t") for papers_tsv in PAPERS_TSVS])

# Drop the papers where can_access is False.
truth_papers = truth_papers[truth_papers.can_access]

# %% Make an NCBI client

# Note this will use the cached web client.
ncbi_client = DiContainer().create_instance(spec={"di_factory": "lib/config/objects/ncbi.yaml"}, resources={})


# %% Load in the query yamls.

query_yamls = [
    "lib/config/queries/mgttest_subset.yaml",
    "lib/config/queries/mgttrain_subset.yaml",
]

queries = [entry for query_yaml in query_yamls for entry in yaml.safe_load(open(query_yaml, "r"))]

# %% For each query, fetch the papers from pubmed, and ask whether every paper in the truth set is returned in that list.

total_missing = 0
total_to_process = 0

for query in queries:
    retmax = query.get("retmax")
    params = {
        "query": query["gene_symbol"],
        "mindate": query.get("min_date"),
        "maxdate": query.get("max_date"),
        # "retmax": query.get("retmax"),
        "retmax": retmax,
    }
    ncbi_pmids = ncbi_client.search(**params)

    if len(ncbi_pmids) == retmax:
        print(f"Warning: Reached retmax for {query['gene_symbol']}.")

    total_to_process += len(ncbi_pmids)

    truth_pmids = truth_papers[truth_papers.gene == query["gene_symbol"]].pmid.astype(str).to_list()

    if missing_from_ncbi := set(truth_pmids) - set(ncbi_pmids):
        print(f"Query: {query['gene_symbol']}")
        print(f"  Got {len(ncbi_pmids)} papers from NCBI.")
        print(f"  Got {len(truth_pmids)} papers from the truth set.")
        print(f"  Missing from NCBI ({query['gene_symbol']}): {missing_from_ncbi}")
        total_missing += len(missing_from_ncbi)
    else:
        pass
        # print(f"  All papers in truth set are available from NCBI for {query['gene_symbol']}.")

print(f"Total papers to process: {total_to_process}")
print(f"Total missing papers: {total_missing}")

# %%
