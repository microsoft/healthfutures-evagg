import requests

from lib.config import PydanticYamlModel


class PubMedAPIConfig(PydanticYamlModel):
    email: str


class PubMedAPI:
    def __init__(self, config: PubMedAPIConfig) -> None:
        # if using Entrez, set the email address here
        # via Entrez.email = config.email
        pass

    def _fetch_paper_bioc(self, pmid: str) -> dict:
        """Fetch a paper from PubMed Central using the BioC API."""
        r = requests.get(
            f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmid}/ascii", timeout=10
        )
        d = r.json()["documents"][0]
        return {"abstract": _get_abstract(d), "title": _get_title(d), "citation": _get_authors(d), "id": pmid}

    def fetch_paper(self, pmid: str) -> dict:
        # TODO, check whether it's in the OA subset?
        return self._fetch_paper_bioc(pmid)


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
