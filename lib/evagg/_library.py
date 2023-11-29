import csv
import json
import os
import re
import xml.etree.ElementTree as Et
from collections import defaultdict
from functools import cache
from typing import Any, Dict, List, Sequence, Set

import requests

from lib.evagg.types import IPaperQuery, Paper, Variant
from lib.evagg.web.entrez import IEntrezClient

from ._interfaces import IGetPapers


class SimpleFileLibrary(IGetPapers):
    def __init__(self, collections: Sequence[str]) -> None:
        self._collections = collections

    def _load_collection(self, collection: str) -> Dict[str, Paper]:
        papers = {}
        # collection is a local directory, get a list of all of the json files in that directory
        for filename in os.listdir(collection):
            if filename.endswith(".json"):
                # load the json file into a dict and append it to papers
                with open(os.path.join(collection, filename), "r") as f:
                    paper = Paper(**json.load(f))
                    papers[paper.id] = paper
        return papers

    @cache
    def _load(self) -> Dict[str, Paper]:
        papers = {}
        for collection in self._collections:
            papers.update(self._load_collection(collection))

        return papers

    def search(self, query: IPaperQuery) -> Set[Paper]:
        # Dummy implementation that returns all papers regardless of query.
        all_papers = set(self._load().values())
        return all_papers


# These are the columns in the truthset that are specific to the paper.
TRUTHSET_PAPER_KEYS = ["doi", "pmid", "pmcid", "paper_title", "link", "is_pmc_oa"]
# These are the columns in the truthset that are specific to the variant.
TRUTHSET_VARIANT_KEYS = [
    "gene",
    "HGVS.C",
    "HGVS.P",
    "phenotype",
    "variant_inheritance",
    "condition_inheritance",
    "study_type",
    "functional_info",
    "mutation_type",
    "notes",
]


class TruthsetFileLibrary(IGetPapers):
    def __init__(self, file_path: str) -> None:
        self._file_path = file_path

    @cache
    def _load_truthset(self) -> Set[Paper]:
        # Group the rows by paper ID.
        paper_groups = defaultdict(list)
        with open(self._file_path) as tsvfile:
            header = [h.strip() for h in tsvfile.readline().split("\t")]
            reader = csv.reader(tsvfile, delimiter="\t")
            for line in reader:
                fields = dict(zip(header, [field.strip() for field in line]))
                paper_id = fields.get("doi") or fields.get("pmid") or fields.get("pmcid") or "MISSING_ID"
                paper_groups[paper_id].append(fields)

        papers: Set[Paper] = set()
        for paper_id, rows in paper_groups.items():
            if paper_id == "MISSING_ID":
                print(f"WARNING: skipped {len(rows)} rows with no paper ID.")
                continue

            # For each paper, extract the paper-specific key/value pairs into a new dict.
            # These are repeated on every paper/variant row, so we can just take the first row.
            paper_data = {k: v for k, v in rows[0].items() if k in TRUTHSET_PAPER_KEYS}

            # Integrity checks.
            for row in rows:
                # Doublecheck if every row has the same values for the paper-specific keys.
                for key in TRUTHSET_PAPER_KEYS:
                    if paper_data[key] != row[key]:
                        print(f"WARNING: multiple values ({paper_data[key]} vs {row[key]}) for {key} ({paper_id}).")
                # Make sure the gene/variant columns are not empty.
                if not row["gene"] or not row["HGVS.P"]:
                    print(f"WARNING: missing gene or variant for {paper_id}.")

            # For each paper, extract the variant-specific key/value pairs into a new dict of dicts.
            variants = {Variant(r["gene"], r["HGVS.P"]): {k: r.get(k, "") for k in TRUTHSET_VARIANT_KEYS} for r in rows}
            # Create a Paper object with the extracted fields.
            papers.add(Paper(id=paper_id, evidence=variants, **paper_data))

        return papers

    def search(self, query: IPaperQuery) -> Set[Paper]:
        all_papers = self._load_truthset()
        query_genes = {v.gene for v in query.terms()}

        # Filter to just the papers with variant terms that have evidence for the genes specified in the query.
        return {p for p in all_papers if query_genes & {v.gene for v in p.evidence.keys()}}


class PubMedFileLibrary(IGetPapers):
    """A class for retrieving papers from PubMed."""

    def __init__(self, entrez_client: IEntrezClient, max_papers: int = 5) -> None:
        """Initialize a new instance of the PubMedFileLibrary class.

        Args:
            entrez_client (IEntrezClient): A class for interacting with the Entrez API.
            max_papers (int, optional): The maximum number of papers to retrieve. Defaults to 5.
        """
        self._entrez_client = entrez_client
        self._max_papers = max_papers

    def search(self, query: IPaperQuery) -> Set[Paper]:
        """Search for papers based on the given query.

        Args:
            query (IPaperQuery): The query to search for.

        Returns:
            Set[Paper]: The set of papers that match the query.
        """
        if len(query.terms()) > 1:
            raise NotImplementedError("Multiple term extraction not yet implemented.")

        term = str(list(query.terms())[0]).split(":")[0]  # TODO: modify to ensure we can extract multiple genes

        id_list = self._find_ids_for_gene(query=term)
        return self._build_papers(id_list)

    def _find_ids_for_gene(self, query: str) -> List[str]:
        id_list_xml = self._entrez_client.esearch(
            db="pmc", sort="relevance", retmax=self._max_papers, retmode="xml", term=query
        )
        tree = Et.fromstring(id_list_xml)
        if (id_list_elt := tree.find("IdList")) is None:
            return []
        id_list = [c.text for c in id_list_elt.iter("Id") if c.text is not None]
        return id_list

    def _fetch_parse_xml(self, id_list: Sequence[str]) -> list:
        ids = ",".join(id_list)

        response = self._entrez_client.efetch(db="pmc", retmode="xml", rettype=None, id=ids)
        root = Et.fromstring(response)

        # Find all 'article' elements
        articles = root.findall("article")

        return articles

    def _find_pmid_in_xml(self, article_elements: List[Any]) -> list:
        list_pmids = []
        for article in article_elements:
            pub_id_elements = article.iter("article-id")
            for pub_id in pub_id_elements:
                if pub_id.get("pub-id-type") == "pmid":
                    list_pmids.append(pub_id.text)
        return list_pmids  # returns PMIDs

    def _get_abstract_and_citation(self, pmid: str) -> tuple:
        text_info = self._entrez_client.efetch(db="pubmed", id=pmid, retmode="text", rettype="abstract")
        citation, doi, pmcid_number = self._generate_citation(text_info)
        abstract = self._extract_abstract(text_info)
        return (citation, doi, abstract, pmcid_number)

    def _generate_citation(self, text_info: str) -> tuple:
        # Extract the author's last name
        match = re.search(r"(\n\n[^\.]*\.)\n\nAuthor information", text_info, re.DOTALL)
        if match:
            sentence = match.group(1).replace("\n", " ")
            author_lastname = sentence.split()[0]
        else:
            author_lastname = None

        # Extract year of publication
        match = re.search(r"\. (\d{4}) ", text_info)
        if match:
            year = match.group(1)
        else:
            year = None

        # Extract journal abbreviation
        match = re.search(r"\. ([^\.]*\.)", text_info)
        if match:
            journal_abbr = match.group(1).strip(".")
        else:
            journal_abbr = None

        # Extract DOI number for citation, TODO: modify to pull key
        match = re.search(r"\nDOI: (.*)\nPMID", text_info)
        match2 = re.search(r"\nDOI: (.*)\nPMCID", text_info)  # TODO: consider embedding into elif
        if match:
            doi_number = match.group(1).strip()
        elif match2:
            doi_number = match2.group(1).strip()
        else:
            doi_number = None

        # Extract PMCID
        match = re.search(r"\nPMCID: (.*)\nPMID", text_info)

        if match:
            pmcid_number = match.group(1).strip()
        else:
            pmcid_number = 0.0

        # Construct citation
        citation = f"{author_lastname} ({year}), {journal_abbr}., {doi_number}"

        return citation, doi_number, pmcid_number

    def _extract_abstract(self, text_info: str) -> str:
        # Extract paragraph after "Author information:" sentence and before "DOI:"
        match = re.search(r"Author information:.*?\.(.*)DOI:", text_info, re.DOTALL)
        if match:
            abstract = match.group(1).strip()
        else:
            abstract = ""
        return abstract

    def _is_pmc_oa(self, pmcid: str) -> bool:
        url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        root = Et.fromstring(response.text)
        if root.find("error") is not None:
            error = root.find("error")
            if error.attrib["code"] == "idIsNotOpenAccess":  # type: ignore
                return False
            else:
                raise NotImplementedError(f"Unexpected error code {error.attrib['code']}")  # type: ignore
        match = next(record for record in root.find("records") if record.attrib["id"] == pmcid)  # type: ignore
        if match:
            return True
        else:
            raise ValueError(f"PMCID {pmcid} not found in response, but records were returned.")

    def _build_papers(self, id_list: list[str]) -> Set[Paper]:  # Dict[str, Dict[str, str]], #3
        papers_tree = self._fetch_parse_xml(id_list)
        list_pmids = self._find_pmid_in_xml(papers_tree)

        # Generate a set of Paper objects
        papers_set = set()
        count = 0
        for pmid in list_pmids:
            citation, doi, abstract, pmcid = self._get_abstract_and_citation(pmid)
            is_pmc_oa = self._is_pmc_oa(pmcid)
            count += 1
            print(count, " Citation: ", citation)
            paper = Paper(
                id=doi, citation=citation, abstract=abstract, pmid=pmid, pmcid=pmcid, is_pmc_oa=is_pmc_oa
            )  # make a new Paper object for each entry
            papers_set.add(paper)  # add Paper object to set

        return papers_set
