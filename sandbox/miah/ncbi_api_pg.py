# Example efetch url
# https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=snp&id=28937887&retmode=json&rettype=json&tool=biopython&email=<REDACTED>&api_key=<REDACTED>

# %% Imports.

import json
import os

import requests
from Bio import Entrez
from defusedxml import ElementTree
from dotenv import load_dotenv

from lib.evagg.web.entrez import BioEntrezClient, BioEntrezDotEnvConfig

# %% Try to reproduce the 400 Error we were receiving.

RSIDS = [
    28937887,
    568256888,
    776518524,
    863223921,
    760988188,
    121918514,
    307955,
    778652037,
    386134164,
    115832790,
    386134166,
    121918518,
    121918514,
    1238280713,
    923331350,
    386134160,
    199779694,
    150345688,
    146010120,
    191999971,
]

# %%


def raw_call(id: str) -> str:
    root = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    args = {
        "db": "snp",
        "id": id,
        "retmode": "json",
        "rettype": "json",
        "tool": "biopython",
        "email": os.getenv("NCBI_EUTILS_EMAIL"),
        "api_key": os.getenv("NCBI_EUTILS_API_KEY"),
    }
    args_str = "&".join([f"{k}={v}" for k, v in args.items()])
    url = f"{root}?{args_str}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text


def entrez_call(id: str, init: bool) -> str:
    if init:
        Entrez.email = os.getenv("NCBI_EUTILS_EMAIL")
        Entrez.max_tries = 1
        # Entrez.api_key = os.getenv("NCBI_EUTILS_API_KEY")

    handle = Entrez.efetch(db="snp", id=id, retmode="json", rettype="json")
    return handle.read()


# %%

for i, id in enumerate(RSIDS):
    init = i == 0
    print(f"{i}: {entrez_call(str(id), init)}")

# %%


bec = BioEntrezClient(BioEntrezDotEnvConfig())

result = bec.efetch(db="snp", id="28937887,568256888", retmode="json", rettype="json")
# might need to -- result = result.decode("utf-8")

splits = result.find("}{")
result0 = result[: splits + 1]
result1 = result[splits + 1 :]

print(json.dumps(json.loads(result0), indent=4))
print(json.dumps(json.loads(result1), indent=4))

# %%
