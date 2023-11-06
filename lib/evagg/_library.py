import csv, json, os, re
from collections import defaultdict
from functools import cache
from typing import Dict, Sequence, Set
from Bio import Entrez # Biopython
import xml.etree.ElementTree as ET

from lib.evagg.types import IPaperQuery, Paper, Variant

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
TRUTHSET_PAPER_KEYS = ["doi", "pmid", "pmcid", "paper_title", "link"]
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
        # Filter to just the papers with variant terms that have evidence
        return {p for p in all_papers if query.terms() & p.evidence.keys()}

from Bio import Entrez
import re
from typing import List

class PubMedFileLibrary(IGetPapers): # TODO: consider gene:variant info next
    
    def __init__(self, email: str, max_papers: int = 5) -> None:
        self._email = email
        Entrez.email = email
        self._max_papers = max_papers
    
    def search_pubmed(self, query: IPaperQuery) -> Set[Paper]: # 1
        term = query.terms()[0]
        id_list = self._find_ids_for_gene(gene=term[:term.find(":")])
        return self._build_papers(id_list)
    
    def _find_ids_for_gene(self, query): # 2
        handle = Entrez.esearch(db='pmc', 
                                sort='relevance', 
                                retmax= self._max_papers,
                                retmode='xml', 
                                term=query)
        id_list = Entrez.read(handle)['IdList']
        return id_list

    def _fetch_parse_xml(self, id_list: Sequence[str]) -> Dict[str, Dict[str, str]]:
        ids = ','.join(id_list)
        handle = Entrez.efetch(db='pmc',
                            retmode='xml',
                            id=ids)
        #results = Entrez.read(handle)
        tree = ET.parse(handle)
        root = tree.getroot()
        all_tree_elements = list(root.iter())

        # list_dois = []
        # for elem in all_tree_elements:
        #     if((elem.tag == "pub-id") and ("/" in str(elem.text))==True):
        #         list_dois.append(elem.text)
        # return(list_dois, all_tree_elements) # returns DOI, XML
        return(all_tree_elements)
    
    def _find_pmid_in_xml(self, all_tree_elements):
        list_pmids = []
        for elem in all_tree_elements:
            if((elem.tag == "pub-id") and ("/" not in str(elem.text))==True): # doi
                list_pmids.append(elem.text)
        return(list_pmids) # returns PMIDs
            
    def _get_abstract_and_citation(self, pmid):
        handle = Entrez.efetch(db='pubmed', id=pmid, retmode='text', rettype='abstract')
        text_info = handle.read()
        citation, doi, pmid = self._generate_citation(text_info)
        abstract = self._extract_abstract(text_info)
        return(citation, doi, pmid, abstract)

    def _generate_citation(self, text_info):
        # Extract the author's last name
        match = re.search(r'(\n\n[^\.]*\.)\n\nAuthor information', text_info, re.DOTALL)
        sentence = match.group(1).replace('\n', ' ')
        author_lastname = sentence.split()[0]
        
        # Extract year and journal abbr.
        year = re.search(r'\. (\d{4}) ', text_info).group(1) # Extract pub. year
        journal_abbr = re.search(r'\. ([^\.]*\.)', text_info).group(1).strip(".") # Extract journal abbreviation
        
        # Extract DOI number for citation, TODO: modify to pull key
        match = re.search(r'\nDOI: (.*)\nPMID', text_info)
        match2 = re.search(r'\nDOI: (.*)\nPMCID', text_info) # TODO: consider embedding into elif
        if match:
            doi_number = match.group(1).strip()
        elif match2:
            doi_number = match2.group(1).strip()
        else:
            doi_number = 0.0
        
        # Extract PMID
        match = re.search(r"PMID: (\d+)", text_info)
        if match:
            pmid_number = match.group(1)
        else:
            pmid_number = 0.0
        
        # Construct citation
        citation = f"{author_lastname} ({year}), {journal_abbr}., {doi_number}"
        
        return citation, doi_number, pmid_number

    def _extract_abstract(self, text_info):
        # Extract paragraph after "Author information:" sentence and before "DOI:"
        abstract = re.search(r'Author information:.*?\.(.*)DOI:', text_info, re.DOTALL).group(1).strip()    
        return abstract

    def _build_papers(self, id_list): # 3
        #papers: List[Paper] = []
        papers_tree = self._fetch_parse_xml(id_list)
        list_pmids = self._find_pmid_in_xml(papers_tree)
        papers_dict = {}
        for pmid in list_pmids:
            citation, doi, pmid, abstract = self._get_abstract_and_citation(pmid)
            papers_dict[doi] = {'abstract': abstract, 'citation': citation, 'pmid': pmid}
        
         # For each paper (DOI), we want to know PMCID, Abstract, Citation.
        # return_dict = {}
        # for result in results:
        #     return_dict[result.doi] = {
        #         "abstract": result.abstract,
        #         "citation": result.citation,
        #         "pmcid": result.pmcid
        #     }
        # # TODO: return a dict key:"DOI", dicts everything else about the paper ()
        # # those would get passed to the contructor of paper
        
        # for key, value in papers_xml.values():
        #     papers.append(Paper(id=key, **value))
        return papers_dict
        
    
class HanoverFileLibrary(IGetPapers):
    
    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
    
    def search(self, query: IPaperQuery) -> Set[Paper]:
        return None
    