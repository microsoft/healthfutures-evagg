# %% Pre-requisites.
# Before running this script you will need to localize the spreadsheet to your local machine.
# Currently there is a copy of this spreadsheet located at https://$SA.blob.core.windows.net/tmp/variant_examples.tsv
# It can be localized with the following commands:
# mkdir -p -m 0777 /mnt/data
# azcopy login
# azcopy cp "https://$SA.blob.core.windows.net/tmp/variant_examples.tsv" "/mnt/data/variant_examples.tsv"

from typing import Tuple

import openpyxl

# %% Imports.
import pandas as pd

# %% Constants.
SPREADSHEET_PATH = "/mnt/data/variant_examples.xlsx"

# %% Read in the spreadsheet as a dataframe.
df = pd.read_excel(SPREADSHEET_PATH, sheet_name="user study variants")

# Pandas loses the hyperlinks for papers, which we need, so we'll use the openpyxl library
# directly to recover them.

wb = openpyxl.load_workbook(SPREADSHEET_PATH)
ws = wb["user study variants"]
link_index = 7

links: dict[Tuple, str | None] = {}

for row in ws.iter_rows():
    if not row[1].value or not row[2].value:
        continue
    key = (row[1].value, row[2].value)
    if row[link_index].hyperlink:
        links[key] = row[link_index].hyperlink.target  # type: ignore

# %% Add the links to the dataframe.


# %% Fetch the papers from the PMC API and generate json representations of each paper.
print("hello")

# %% Fetch an equivalent number of negative controls.

# %% Construct each dataset as a collection of these json files and upload to the corresponding cloud storage location.
