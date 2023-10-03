import json

from Bio import Entrez
from Bio.Entrez.Parser import DictionaryElement

Entrez.email = "miah@microsoft.com"

# %% Constants.
PMIDS = ["31453292", "34758253"]


# %% Functions.


def fetch_paper_entrez(pmid: str) -> dict:
    """Fetch a paper from PubMed.

    Note that the Entrez API can only retrieve abstracts for PMC papers.
    """
    handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
    contents = Entrez.read(handle)
    handle.close()
    if not isinstance(contents, DictionaryElement):
        raise TypeError("Expected a DictionaryElement.")

    return {
        "Abstract": contents["PubmedArticle"][0]["MedlineCitation"]["Article"]["Abstract"]["AbstractText"][0],
        "Title": contents["PubmedArticle"][0]["MedlineCitation"]["Article"]["ArticleTitle"],
        "Authors": contents["PubmedArticle"][0]["MedlineCitation"]["Article"]["AuthorList"],
    }


paper = fetch_paper_entrez(PMIDS[0])
print(json.dumps(paper, indent=4))
