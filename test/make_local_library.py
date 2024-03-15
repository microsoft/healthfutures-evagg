"""This notebook generates a local document library intended to support lib.evagg.SimpleFileLibrary."""

# %% Imports.
import json
import os

import requests

# %% Constants.

# These PMIDS are drawn from variant_examples.xlsx.
PMIDS = ["31453292", "34758253"]
library_path = ".data/evagg_local"


# %% Fetch the full paper text using BioC.


def _get_doi(d: dict) -> str:
    return d["passages"][0]["infons"]["article-id_doi"]


def _get_abstract(d: dict) -> str:
    candidates = [e["text"] for e in d["passages"] if e["infons"]["type"] == "abstract"]
    if len(candidates) == 1:
        return candidates[0]
    else:
        return "???"


def _get_title(d: dict) -> str:
    candidates = [e["text"] for e in d["passages"] if e["infons"]["type"] == "title"]
    if len(candidates) == 1:
        return candidates[0]
    else:
        return "???"


def _fix(s: str) -> str:
    return s.replace("surname:", "").replace(";", ", ").replace("given-names:", "").strip()


def _get_authors(d: dict) -> str:
    candidates = [e["infons"] for e in d["passages"] if e["infons"]["section_type"] == "TITLE"]
    if len(candidates) == 1:
        candidate: dict = candidates[0]
        authors = "; ".join([_fix(candidate[k]) for k in candidate.keys() if k.startswith("name_")])
        return authors
    else:
        return "???"


def _get_pmcid(d: dict) -> str:
    return f"PMC{d['id']}"


def fetch_paper_bioc(pmid: str) -> dict:
    """Fetch a paper from PubMed Central using the BioC API."""
    r = requests.get(
        f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmid}/ascii", timeout=10
    )
    r.raise_for_status()
    d = r.json()[0]["documents"][0]
    return {
        "id": _get_doi(d),
        "abstract": _get_abstract(d),
        "title": _get_title(d),
        "citation": _get_authors(d),
        "pmcid": _get_pmcid(d),
    }


# %% Generate the library of PMIDs.

# Make sure library_path exists, create if it doesn't.
if not os.path.exists(library_path):
    os.makedirs(library_path)

for id in PMIDS:
    paper = fetch_paper_bioc(id)
    print(f"Writing {id} to {library_path}/{id}.json")
    with open(f"{library_path}/{id}.json", "w") as f:
        json.dump(paper, f, indent=4)
