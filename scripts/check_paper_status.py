"""Script to check the status of a paper in pubmed."""

# %% Imports.

import datetime
import re
from functools import cache
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from lib.di import DiContainer
from lib.evagg.ref import IPaperLookupClient

# %% Constants.

PMIDS = {
    "BANF1": [
        "21549337",
        "32783369",
        "33660778",
        "36039758",
    ],
    "BCL9": [
        "32368696",
        "33084842",
    ],
    "CACNA1D": ["21131953", "30498240", "32747562", "10929716", "15357422", "21131953", "22678062"],
    "CCDC50": [
        "17503326",
        "24875298",
        "27068579",
        "27911912",
        "33229591",
        "34416374",
        "37165931",
        "17503326",
        "37165931",
    ],
    "FANCM": ["26130695", "31700994", "33471991", "36707629", "31700994"],
    "KLF13": ["32293321", "33215447", "35369534", "17053787", "28164238"],
    "LBR": ["12618959", "18382993", "21327084", "30518689", "30561119", "21327084"],
    "MBD4": ["30049810", "35381620", "35460607", "17285135", "27086921", "37957685"],
    "TCIRG1": ["30537558"],
}

# %% Functions.


@cache
def get_ncbi_client() -> IPaperLookupClient:
    return DiContainer().create_instance(spec={"di_factory": "lib/config/objects/ncbi.yaml"}, resources={})


@cache
def get_pmid_list_all(gene_symbol: str, retmax: int = 10000) -> Sequence[str]:
    """Get the list of PMIDs for a given gene symbol."""
    ncbi_client = get_ncbi_client()
    print(f"Querying all of pubmed for {gene_symbol}...")
    params = {"query": gene_symbol, "retmax": retmax}
    pmids = ncbi_client.search(**params)
    if len(pmids) == retmax:
        print(
            f"Warning: when querying all of pubmed for {gene_symbol}, got {retmax} results. This may be an incomplete result."
        )
    return pmids


@cache
def load_gene_configs() -> Dict[str, Dict[str, Any]]:
    config_file = "lib/config/queries/clingen_10_genes.yaml"
    with open(config_file, "r") as f:
        config_raw = yaml.safe_load(f)

    config: Dict[str, Dict[str, Any]] = {}

    for elt in config_raw:
        if "gene_symbol" not in elt:
            continue
        config[elt.pop("gene_symbol")] = elt

    return config


@cache
def get_pmid_list_query(gene_symbol: str) -> Sequence[str]:
    """A smarter version of the above function that uses a config to get a list that's representative of what the
    pipeline would actually have to work with."""

    gene_configs = load_gene_configs()
    query = gene_configs.get(gene_symbol, {})

    params: dict[str, Any] = {"query": f"{gene_symbol} pubmed pmc open access[filter]"}
    # Rationalize the optional parameters.
    if ("max_date" in query or "date_type" in query) and "min_date" not in query:
        raise ValueError("A min_date is required when max_date or date_type is provided.")
    if "min_date" in query:
        params["mindate"] = query["min_date"]
        params["date_type"] = query.get("date_type", "pdat")
        params["maxdate"] = query.get("max_date", datetime.datetime.now().strftime("%Y/%m/%d"))
    if "retmax" in query:
        params["retmax"] = query["retmax"]

    return get_ncbi_client().search(**params)


def find_line_in_logs(query: str) -> str:
    """Search through every line in the log files for a line that matches query. Return the first match."""
    log_files = [
        ".out/run_zevagg_pipeline_clingen_10_genes_20250305_145143/console.log",
        ".out/run_zevagg_pipeline_clingen_10_genes_20250305_122938/console.log",
        # ".out/run_evagg_pipeline_20250317_191428/console.log",
        # ".out/run_evagg_pipeline_20250317_170658/console.log",
    ]
    for log_file in log_files:
        with open(log_file, "r") as f:
            for line in f:
                if query in line:
                    return line
    return ""


def get_paper_dict(pmid: str, gene_symbol: str) -> Dict[str, Any]:
    """Get the paper dictionary for a given pmid."""
    ncbi_client = get_ncbi_client()
    paper = ncbi_client.fetch(pmid, include_fulltext=True)

    paper_dict: Dict[str, Any] = {}
    paper_dict["pmid"] = pmid
    paper_dict["gene_symbol"] = gene_symbol
    paper_dict["status"] = "OOPS - unknown status"
    if not paper:
        paper_dict["status"] = "fetch error"
    else:
        paper_dict["title"] = paper.props.get("title", None) or None
        paper_dict["journal"] = paper.props.get("journal", None) or None
        paper_dict["doi"] = paper.props.get("doi", None) or None
        paper_dict["pmcid"] = paper.props.get("pmcid", None) or None
        paper_dict["oa"] = paper.props.get("OA", None) or None
        paper_dict["license"] = paper.props.get("license", None) or None
        paper_dict["has_fulltext"] = paper.props.get("fulltext_xml", "") != ""
        # Decode status based on the above.
        # if pmid not in get_pmid_list_query(gene_symbol):
        #     paper_dict["status"] = "not returned by pubmed OA search"
        if pmid not in get_pmid_list_all(gene_symbol):
            paper_dict["status"] = "not returned by pubmed general search"
        elif not paper_dict["pmcid"]:
            paper_dict["status"] = "not in PMC"
        elif paper_dict["license"] == "not_open_access":
            paper_dict["status"] = "not OA"
        elif paper_dict["license"] and "ND" in paper_dict["license"]:
            paper_dict["status"] = "bad license"
        else:
            if line := find_line_in_logs(f"observations in pmid:{pmid} for {gene_symbol}"):
                match = re.search(r"\d+ observations", line)
                if match:
                    paper_dict["status"] = match.group(0)
                else:
                    paper_dict["status"] = "OOPS - error finding obs count"
            elif line := find_line_in_logs(f"No observations found in pmid:{pmid} for {gene_symbol}"):
                paper_dict["status"] = "No observations"

    return paper_dict


# %% Iterate over all PMIDs, building a dataframe.

paper_dicts: List[Dict[str, Any]] = []

for gene_symbol, pmids in PMIDS.items():
    for pmid in set(pmids):
        paper_dict = get_paper_dict(pmid, gene_symbol)
        paper_dicts.append(paper_dict)

df = pd.DataFrame(paper_dicts)

# %% Make some plots.

df["status_clean"] = df["status"]

# replace all instances of status_clean that are \d observations where \d is greater than 0 with "found observations"
df["status_clean"] = df["status_clean"].replace(
    to_replace=r"(\d+) observations",
    value=">=1 variants",
    regex=True,
)

df["status_clean"] = df["status_clean"].replace(
    to_replace="No observations",
    value="0 variants",
)

# replace "OOPS - unknown status" with "paper finding miss"
df["status_clean"] = df["status_clean"].replace(
    to_replace="OOPS - unknown status",
    value="paper finding miss",
)

df["status_clean"] = df["status_clean"].replace(
    to_replace="bad license",
    value="unusable license",
)

# Reorder the 'status_clean' column based on value counts (tallest to shortest)
status_order = df["status_clean"].value_counts().index
df["status_clean"] = pd.Categorical(df["status_clean"], categories=status_order, ordered=True)

# Plot a histogram of the status column, set the order of columns to be descending by count
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(
    df,
    x="status_clean",
    discrete=True,
    shrink=0.8,
)

# Set the x-axis label rotation angle to 90 degrees
plt.xticks(rotation=90)

# Set the xlabel to paper status
plt.xlabel("Paper Status")
# Set the title to "EvAgg Missing PMID Causes"
plt.title("EvAgg Missing PMID Causes")
print(df.status_clean.value_counts())

# %%
