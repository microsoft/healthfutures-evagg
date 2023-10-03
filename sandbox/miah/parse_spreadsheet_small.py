# %% Pre-requisites.
# Before running this script you will need to localize the spreadsheet to your local machine.
# Currently there is a copy of this spreadsheet located at https://$SA.blob.core.windows.net/tmp/variant_examples.xlsx
# It can be localized with the following commands:
# sudo mkdir -p -m 0777 /mnt/data
# azcopy login
# azcopy cp "https://$SA.blob.core.windows.net/tmp/variant_examples.xlsx" "/mnt/data/variant_examples.xlsx"

from typing import Tuple

import openpyxl

# %% Imports.
import pandas as pd

# %% Constants.
SPREADSHEET_PATH = "/mnt/data/variant_examples.xlsx"

# %% Read in the spreadsheet as a dataframe.
df = pd.read_excel(SPREADSHEET_PATH, sheet_name="user study variants")

# Quick cleanup of spreadsheet, remove all empty rows. Remove all rows corresponding to Family IDs.
df = df.dropna(how="all")
df = df[not df["Gene"].str.startswith("RGP_")]

# %% Add the links into the dataframe (they're lost by pandas).

# Pandas loses the hyperlinks for papers, which we need, so we'll use the openpyxl library
# directly to recover them.

wb = openpyxl.load_workbook(SPREADSHEET_PATH)
ws = wb["user study variants"]
link_index = 7

links: dict[Tuple, str | None] = {}

for row in ws.iter_rows():
    if not row[0].value or not row[1].value:
        continue
    key = (row[0].value, row[1].value)
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

# %% Extract the PMCID from the links.
df["pmcid"] = df["link"].str.extract(r"https://pubmed.ncbi.nlm.nih.gov/(\d+)")
