"""Notebook to explore whether/which all papers in the truth set are available to us via pubmed search."""

# %% Imports.

import pandas as pd
import yaml

from lib.di import DiContainer

# %% Constants.

PAPERS_TSVS = ["data/v1/papers_test_v1.tsv", "data/v1/papers_train_v1.tsv"]

# %% Load in the truthset papers.

truth_papers = pd.concat([pd.read_csv(papers_tsv, sep="\t") for papers_tsv in PAPERS_TSVS])

# Drop the papers where can_access is False.
truth_papers = truth_papers[truth_papers.can_access]

# %% Make an NCBI client

# Note this will use the cached web client.
ncbi_client = DiContainer().create_instance(spec={"di_factory": "lib/config/objects/ncbi.yaml"}, resources={})

# %% Define helper functions.


def get_ncbi_papers_query_only(query, ncbi_client, warn_on_reach_retmax=False):
    params = {
        "query": f"Gene {query['gene_symbol']} pubmed pmc open access[filter]",
        "mindate": query.get("min_date"),
        "maxdate": query.get("max_date"),
        "retmax": query.get("retmax"),
    }
    ncbi_pmids = ncbi_client.search(**params)

    if warn_on_reach_retmax and len(ncbi_pmids) == query.get("retmax"):
        print(f"Warning: Reached retmax for {query['gene_symbol']}.")

    return ncbi_pmids


def get_ncbi_papers_query_and_filter(query, ncbi_client, warn_on_reach_retmax=False):
    params = {
        "query": query["gene_symbol"],
        "mindate": query.get("min_date"),
        "maxdate": query.get("max_date"),
        "retmax": query.get("retmax"),
    }
    ncbi_pmids = ncbi_client.search(**params)

    if warn_on_reach_retmax and len(ncbi_pmids) == query.get("retmax"):
        print(f"Warning: Reached retmax for {query['gene_symbol']}.")

    kept_pmids = []
    for pmid in ncbi_pmids:
        paper = ncbi_client.fetch(pmid, include_fulltext=False)
        if paper and paper.props.get("OA", False) is True:
            kept_pmids += [pmid]

    return kept_pmids


# %% Load in the query yamls.

query_yamls = [
    "lib/config/queries/mgttest_subset.yaml",
    "lib/config/queries/mgttrain_subset.yaml",
]

query_tuples = [
    (entry, "test" if query_yaml.find("mgttest") > 0 else "train")
    for query_yaml in query_yamls
    for entry in yaml.safe_load(open(query_yaml, "r"))
]

# Make a dataframe from query tuples, where one column is the gene symbol and the other is 'train' or 'test'
gene_group_df = pd.DataFrame([(t[0]["gene_symbol"], t[1]) for t in query_tuples], columns=["gene_symbol", "group"])

queries = [t[0] for t in query_tuples]

# %% For each query, fetch papers from pubmed using the two approaches defined above.

# If you want to run from cache, use this line instead.
# all_pmid_df = pd.read_csv("/home/azureuser/ev-agg-exp/.tmp/pmcoa_query_comparison_incl_gene.tsv", sep='\t')

all_query_dfs = []

for query in queries:
    ncbi_pmids_query_only = get_ncbi_papers_query_only(query, ncbi_client)
    ncbi_pmids_query_and_filter = get_ncbi_papers_query_and_filter(query, ncbi_client)

    truth_pmids = truth_papers[truth_papers.gene == query["gene_symbol"]].pmid.astype(str).to_list()

    # Generate a dataframe that contains all the pmids from the truth set and the two pubmed query approaches, retaining
    # whether a given pmid is in the truth set, is returned by the query only approach, and is returned by the query and filter approach.
    all_pmids = list(set(truth_pmids + ncbi_pmids_query_only + ncbi_pmids_query_and_filter))

    query_df = pd.DataFrame(
        {
            "pmid": all_pmids,
            "in_truth_set": [pmid in truth_pmids for pmid in all_pmids],
            "in_query_only": [pmid in ncbi_pmids_query_only for pmid in all_pmids],
            "in_query_and_filter": [pmid in ncbi_pmids_query_and_filter for pmid in all_pmids],
        }
    )
    query_df["gene_symbol"] = query["gene_symbol"]
    query_df["group"] = gene_group_df[gene_group_df.gene_symbol == query["gene_symbol"]].group.values[0]
    all_query_dfs.append(query_df)

# Concatenate all the query dataframes into one.
all_pmid_df = pd.concat(all_query_dfs, ignore_index=True)


# %% Based on the dataframe, let's write out some statistics, assessing how many papers are missing from train/test, and which ones are missing on a per gene basis.

total_missing_query_only = 0
total_missing_query_and_filter = 0

for gene, gene_papers_df in all_pmid_df.groupby("gene_symbol"):
    gene_papers_query_only = gene_papers_df.query("in_query_only == True").pmid.to_list()
    gene_papers_query_and_filter = gene_papers_df.query("in_query_and_filter == True").pmid.to_list()
    truth_papers_for_gene = truth_papers[truth_papers.gene == gene].pmid.astype(int).to_list()

    missing_from_query_only = set(truth_papers_for_gene) - set(gene_papers_query_only)
    missing_from_query_and_filter = set(truth_papers_for_gene) - set(gene_papers_query_and_filter)

    unique_to_query_only = set(gene_papers_query_only) - set(gene_papers_query_and_filter)
    unique_to_query_and_filter = set(gene_papers_query_and_filter) - set(gene_papers_query_only)

    # Gene-specific assessment
    print(f"Gene: {gene} ({gene_group_df.query(f'gene_symbol == "{gene}"').group.values[0]})")
    print(f"  Total papers in truth set:                        {len(truth_papers_for_gene)}")
    print(f"  Papers in query only approach:                    {len(gene_papers_query_only)}")
    print(f"  Papers in query and filter approach:              {len(gene_papers_query_and_filter)}")
    print(f"  Truth papers absent in query only approach:       {len(missing_from_query_only)}")
    print(f"  Truth papers absent in query and filter approach: {len(missing_from_query_and_filter)}")
    print(f"  Unique to query only approach:                    {len(unique_to_query_only)}")
    if len(unique_to_query_only) > 0:
        print(f"    - {unique_to_query_only}")
    print(f"  Unique to query and filter approach:              {len(unique_to_query_and_filter)}")

    # Preparation for global assessment.
    total_missing_query_only += len(missing_from_query_only)
    total_missing_query_and_filter += len(missing_from_query_and_filter)

# Global assessment
print(f"Total missing papers from query only approach: {total_missing_query_only}")
print(f"Total missing papers from query and filter approach: {total_missing_query_and_filter}")

# %% Look at those that were in truth, and only found in one set but not the other.

all_pmid_df.query("in_truth_set == True").query("in_query_only != in_query_and_filter")


# %% Check to see whether the keyword "gene" is present in the fulltext of the papers.

for pmid in all_pmid_df.query("in_truth_set == True").query("in_query_only != in_query_and_filter").pmid:
    paper = ncbi_client.fetch(str(pmid), include_fulltext=True)

    xml = paper.props["fulltext_xml"]
    if any(kwd in xml for kwd in [" gene ", " Gene ", " genes ", " Genes "]):
        print(f"Keyword found in {pmid}")
    else:
        print(f"!! Keyword not found in {pmid}")

# %%
