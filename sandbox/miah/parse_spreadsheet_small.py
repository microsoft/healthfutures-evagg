"""parse_spreadsheet_small.py.

This script takes as input a manually generated spreadsheet of evidence relevant to the assessment of a collection of
gene/variant pairs. It outputs a dataframe containing the same information as the spreadsheet, but in a format that
can be easily consumed by downstream scripts.

The output dataframe will contain the following columns:
- query_gene
- query_variant (Optional)
- gene
- HGVS.C
- HGVS.P
- doi
- pmid
- pmcid
- phenotype
- variant_inheritance
- condition_inheritance
- study_type
- functional_info
- mutation_type
- paper_title
- link
- notes

Note, each row will be uniqely identified by gene, variant, doi. That means if a paper mentions multiple variants,
there will be a separate row in the dataframe for each mention within the paper. If a variant is mentioned in multiple
papers, there will be a separate row in the dataframe for each mention of the variant.
"""
# %% Pre-requisites.
# Before running this script you will need to localize the spreadsheet to your local machine.
# Currently there is a copy of this spreadsheet located at https://$SA.blob.core.windows.net/tmp/variant_examples.xlsx
# It can be localized with the following commands:
# sudo mkdir -p -m 0777 /mnt/data
# azcopy login (only works with azcopy < 10.20, more recent versions need export AZCOPY_AUTO_LOGIN_TYPE=DEVICE)
# azcopy cp "https://$SA.blob.core.windows.net/tmp/variant_examples.xlsx" "/mnt/data/variant_examples.xlsx"
#
# To use the Entrez APIs you'll also need to point CONFIG_PATH to a config file that contains an attribute "email"
# at the root level, e.g.
# email: your_email_address

# %% Imports.

from functools import cache
from typing import Any, Tuple

import numpy as np
import openpyxl
import pandas as pd
import requests
from Bio import Entrez, Medline

from lib.config import PydanticYamlModel

# %% Constants.
SPREADSHEET_PATH = "/mnt/data/variant_examples.xlsx"
CONFIG_PATH = "/home/azureuser/repos/ev-agg-exp/sandbox/miah/.config/parse_spreadsheet_small_config.yaml"
OUTPUT_PATH_TEMPLATE = "/mnt/data/truth_set_SIZE.tsv"
TINY_GENE_SET = ["COQ2", "JPH1"]

# %% Handle configuration.


class Config(PydanticYamlModel):
    email: str


config = Config.parse_yaml(CONFIG_PATH)

# Set up Entrez API.
Entrez.email = config.email

# %% Read in the spreadsheet as a dataframe.
df = pd.read_excel(SPREADSHEET_PATH, sheet_name="user study variants")

# Quick cleanup of spreadsheet, remove all empty rows. Remove all rows corresponding to Family IDs.
df = df.dropna(how="all")
df = df[~df["Gene"].str.startswith("RGP_")]

# Strip whitespace from some string columns.
df["Gene"] = df["Gene"].str.strip()
df["HGVS.C"] = df["HGVS.C"].str.strip()
df["HGVS.P"] = df["HGVS.P"].str.strip()

# %% Add the links into the dataframe (they're lost by pandas).

# Pandas loses the hyperlinks for papers, which we need, so we'll use the openpyxl library
# directly to recover them.

wb = openpyxl.load_workbook(SPREADSHEET_PATH)
ws = wb["user study variants"]
link_index = 7

links: dict[Tuple, str | None] = {}

count = 0
for row in ws.iter_rows():
    if not row[0].value or not row[1].value:
        continue
    key = (str(row[0].value).strip(), str(row[1].value).strip())
    if row[link_index].hyperlink:
        links[key] = row[link_index].hyperlink.target  # type: ignore

# Recover the links
df = pd.merge(
    df,
    pd.Series(links).reset_index(),  # type: ignore
    left_on=["Gene", "HGVS.C"],
    right_on=["level_0", "level_1"],
    how="left",
)
df = df.rename(columns={0: "link"})
df = df.drop(columns=["level_0", "level_1"])

# %% Extract the PMID from the links.
df["pmid"] = df["link"].str.extract(r"https://pubmed.ncbi.nlm.nih.gov/(\d+)")

# %% Populate the PMCID and DOI.


@cache
def _call_converter(ids_cat: str) -> dict[str, Any]:
    service_root = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    email = config.email
    tool = "research_use"
    format = "json"

    url = f"{service_root}?tool={tool}&email={email}&ids={ids_cat}&format={format}"

    response = requests.get(url, timeout=5)
    response.raise_for_status()

    return response.json()


def convert_ids(ids: pd.Series, tgt: str) -> pd.Series | None:
    # Example syntax
    # https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool=my_tool&email=my_email@example.com&ids=PMC1193645
    # See https://www.ncbi.nlm.nih.gov/pmc/tools/id-converter-api/ for API documentation
    max_len = 200

    ids_str = [str(id) for id in set(ids.dropna().tolist())]
    if len(ids_str) == 0:
        return None
    if len(ids_str) > max_len:
        raise ValueError(f"Number of ids to process must be less than {max_len} in length, got {len(ids_str)}")

    raw = _call_converter(",".join(ids_str))
    lookup = {record["pmid"]: record[tgt] for record in raw["records"] if tgt in record}

    return pd.Series(ids.map(lookup))


df["pmcid"] = convert_ids(df["pmid"], "pmcid")
df["doi"] = convert_ids(df["pmid"], "doi")


# %% Build paper_id


# %% Fetch paper title


def fetch_paper_title(pmids: pd.Series) -> pd.Series:
    ids = set(pmids.dropna().tolist())
    handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
    records = Medline.parse(handle)
    lookup = {record.get("PMID", ""): record.get("TI", "") for record in records}
    return pd.Series(pmids.map(lookup))


df["paper_title"] = fetch_paper_title(df["pmid"])

# %% Reformat and reorder dataframe

df["query_gene"] = df["Gene"]
# Currently we don't have a way to specify a query variant, so we'll just leave this column blank.
df["query_variant"] = np.nan
df.rename(
    columns={
        "Gene": "gene",
        "Condition/ Phenotype": "phenotype",
        "Zygosity": "zygosity",
        "Reported Variant Inheritance": "variant_inheritance",
        "Condition Inheritance ": "condition_inheritance",
        "Study type": "study_type",
        "Functional info": "functional_info",
        "Mutation type": "mutation_type",
        "Notes": "notes",
    },
    inplace=True,
)
df = df[
    [
        "query_gene",
        "query_variant",
        "gene",
        "HGVS.C",
        "HGVS.P",
        "doi",
        "pmid",
        "pmcid",
        "phenotype",
        "variant_inheritance",
        "condition_inheritance",
        "study_type",
        "functional_info",
        "mutation_type",
        "paper_title",
        "link",
        "notes",
    ]
]

# %% Write small truth set to TSV
df.to_csv(OUTPUT_PATH_TEMPLATE.replace("SIZE", "small"), index=False, sep="\t")

# %% Write tiny truth set to TSV
df = df[df["gene"].isin(TINY_GENE_SET)]
df.to_csv(OUTPUT_PATH_TEMPLATE.replace("SIZE", "tiny"), index=False, sep="\t")
