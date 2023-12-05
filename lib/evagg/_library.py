import csv
import json
import os
import re
import xml.etree.ElementTree as Et
from collections import defaultdict
from functools import cache
from typing import Dict, List, Sequence, Set, Tuple

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
TRUTHSET_PAPER_KEYS = ["doi", "pmid", "pmcid", "is_pmc_oa", "license", "paper_title", "link"]
# These are the columns in the truthset that are specific to the variant.
TRUTHSET_VARIANT_KEYS = [
    "gene",
    "HGVS.C",
    "HGVS.P",
    "phenotype",
    "zygosity",
    "variant_inheritance",
    "study_type",
    "functional_study",
    "variant_type",
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

        pmid_list = self._find_pmids_for_gene(query=term)
        return self._build_papers(pmid_list)

    def _find_pmids_for_gene(self, query: str) -> List[str]:
        # Search the pubmed database
        pmid_list_xml = self._entrez_client.esearch(db="pubmed", sort="relevance", retmax=self._max_papers, term=query)

        # Extract the IDs
        tree = Et.fromstring(pmid_list_xml)
        if (id_list_elt := tree.find("IdList")) is None:
            return []
        pmid_list = [c.text for c in id_list_elt.iter("Id") if c.text is not None]
        return pmid_list

    def _get_abstract_and_citation(self, pmid: str) -> Tuple[str, str | None, str | None, str | None]:
        xml_info = self._entrez_client.efetch(db="pubmed", id=pmid, retmode="xml", rettype="abstract")
        records = Et.fromstring(xml_info)

        # get abstract
        abstract_elem = records.find(".//AbstractText")
        if abstract_elem is not None:
            abstract = abstract_elem.text
        else:
            abstract = None

        # generate citation
        # get first author last name info
        first_author_elem = records.find(".//Author/LastName")
        if first_author_elem is not None:
            first_author_last_name = first_author_elem.text
        else:
            first_author_last_name = None

        # Extract year of publication
        pub_year_elem = records.find(".//PubDate/Year")
        if pub_year_elem is not None:
            pub_year = pub_year_elem.text
        else:
            pub_year = "0.0"

        # extract journal abbreviation
        journal_abbreviation_elem = records.find(".//ISOAbbreviation")
        if journal_abbreviation_elem is not None:
            journal_abbreviation = journal_abbreviation_elem.text
        else:
            journal_abbreviation = None

        # extract DOI
        doi_elem = records.find(".//ELocationID[@EIdType='doi']")
        if doi_elem is not None:
            doi = doi_elem.text
        else:
            doi = "0.0"

        # extract PMCID
        pmcid_elem = records.find(".//ArticleId[@IdType='pmc']")
        if pmcid_elem is not None:
            pmcid = pmcid_elem.text
        else:
            pmcid = "0.0"

        # generate citation
        citation = f"{first_author_last_name} ({pub_year}) {journal_abbreviation}, {doi}"

        return citation, doi, abstract, pmcid

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

    def _build_papers(self, pmid_list: list[str]) -> Set[Paper]:  # Dict[str, Dict[str, str]]
        # Generate a set of Paper objects
        papers_set = set()
        count = 0
        for pmid in pmid_list:
            citation, doi, abstract, pmcid = self._get_abstract_and_citation(pmid)
            is_pmc_oa = self._is_pmc_oa(pmcid) if pmcid is not None else False
            count += 1
            print(count, " Citation: ", citation)
            paper = Paper(
                id=doi, citation=citation, abstract=abstract, pmid=pmid, pmcid=pmcid, is_pmc_oa=is_pmc_oa
            )  # make a new Paper object for each entry
            papers_set.add(paper)  # add Paper object to set

        return papers_set
