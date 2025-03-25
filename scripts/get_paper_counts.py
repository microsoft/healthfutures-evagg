"""Determine the number of pubmed pmids returned (regardless of PMCOA status) for a config."""

# %% imports.

import yaml

from lib.di import DiContainer
from lib.evagg.ref import IPaperLookupClient

# %% get instance

client: IPaperLookupClient = DiContainer().create_instance(
    spec={"di_factory": "lib/config/objects/ncbi.yaml"}, resources={}
)

# %% get the gene list and date range

with open("lib/config/queries/clingen_10_genes.yaml", "r") as f:
    gene_dict = yaml.safe_load(f)

# %%

for params in gene_dict:
    params["query"] = params.pop("gene_symbol")
    params["mindate"] = params.pop("min_date")
    params["date_type"] = "pdat"
    params["maxdate"] = params.pop("max_date")
    params["retmax"] = 10000

    print(f"{params['query']} - {len(client.search(**params))}")

# %%
