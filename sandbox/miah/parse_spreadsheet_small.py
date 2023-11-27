"""parse_spreadsheet_small.py.

This script takes as input a manually generated spreadsheet of evidence relevant to the assessment of a collection of
gene/variant pairs. It outputs a dataframe containing the same information as the spreadsheet, but in a format that
can be easily consumed by downstream scripts.

The output dataframe will contain the columns described in data/README.md

Note, each row will be uniqely identified by gene, variant, doi. That means if a paper mentions multiple variants,
there will be a separate row in the dataframe for each mention within the paper. If a variant is mentioned in multiple
papers, there will be a separate row in the dataframe for each mention of the variant.
"""

# %% Pre-requisites.
# Before running this script you will need to localize the source spreadsheet to your local machine.
# Currently there is a copy of this spreadsheet located at https://$SA.blob.core.windows.net/tmp/variant_examples.xlsx
# It can be localized with the following commands:
# sudo mkdir -p .data/truth
# azcopy login (only works with azcopy < 10.20, more recent versions need export AZCOPY_AUTO_LOGIN_TYPE=DEVICE)
# azcopy cp "https://$SA.blob.core.windows.net/tmp/variant_examples.xlsx" ".data/truth/variant_examples.xlsx"
#
# To use the Entrez APIs you'll also need to point CONFIG_PATH to a config file that contains an attribute "email"
# at the root level, e.g.
# email: your_email_address

# %% Imports.

import os
from functools import cache
from typing import Any, Tuple

import numpy as np
import openpyxl
import pandas as pd
import requests
from Bio import Entrez, Medline
from defusedxml import ElementTree

from lib.config import PydanticYamlModel

# %% Constants.
# Goofy joins to support both running interactive and as a script.
SPREADSHEET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", ".data", "truth", "variant_examples.xlsx")
WORKSHEET_NAME = "pilot study variants"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), ".config", "parse_spreadsheet_small_config.yaml")
OUTPUT_PATH_TEMPLATE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "truth_set_SIZE.tsv")
TINY_GENE_SET = ["COQ2", "JPH1"]

# %% Handle configuration.


class Config(PydanticYamlModel):
    email: str


config = Config.parse_yaml(CONFIG_PATH)

# Set up Entrez API.
Entrez.email = config.email

# %% Read in the spreadsheet as a dataframe.
df = pd.read_excel(SPREADSHEET_PATH, sheet_name=WORKSHEET_NAME)

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
ws = wb[WORKSHEET_NAME]
link_index = 6

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
    # Note: this will fail to find the article if it's not in PMC.

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


# %% Build doi and paper title


def _parse_id_dict(raw: list[str]) -> dict[str, str]:
    # each identifier is of the form 'value [key]', parse to dict {key: value}
    retval: dict[str, str] = {}

    for line in raw:
        tokens = line.split(" [")
        if len(tokens) == 2:
            retval[tokens[1][:-1]] = tokens[0]

    return retval


ids = set(df["pmid"].dropna().tolist())
handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
records = Medline.parse(handle)
lookup = {record.get("PMID", ""): record.get("TI", "") for record in records}
df["paper_title"] = pd.Series(df["pmid"].map(lookup))

handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
records = Medline.parse(handle)
lookup = {record.get("PMID", ""): _parse_id_dict(record.get("AID", "")).get("doi", None) for record in records}

df["doi"] = pd.Series(df["pmid"].map(lookup))

# %% determine whether the paper is in PMC-OA, only need to do this in cases where there's a PMCID


def _get_oa_info(pmcid: str) -> pd.Series:
    url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
    response = requests.get(url, timeout=5)
    response.raise_for_status()

    root = ElementTree.fromstring(response.text)
    if root.find("error") is not None:
        error = root.find("error")
        if error.attrib["code"] == "idIsNotOpenAccess":  # type: ignore
            return pd.Series({"is_pmc_oa": False, "license": "unknown"})
        else:
            raise ValueError(f"Unexpected error code {error.attrib['code']}")  # type: ignore
    match = next(record for record in root.find("records") if record.attrib["id"] == pmcid)  # type: ignore
    if match:
        license = match.attrib["license"] if "license" in match.attrib else "unknown"
        return pd.Series({"is_pmc_oa": True, "license": license})
    else:
        raise ValueError(f"PMCID {pmcid} not found in response, but records were returned.")


df = pd.concat([df, df["pmcid"].dropna().apply(_get_oa_info)], axis=1)
df.fillna({"is_pmc_oa": False, "license": "unknown"}, inplace=True)

# df["pmcid"].dropna().map(_get_oa_info)

# pd.concat([df, temp_df], axis=1)

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
        "Study type": "study_type",
        "Functional info": "functional_study",
        "Variant type": "variant_type",
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
        "is_pmc_oa",
        "license",
        "phenotype",
        "zygosity",
        "variant_inheritance",
        "study_type",
        "functional_study",
        "variant_type",
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

# %%
