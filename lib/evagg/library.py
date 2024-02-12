import csv
import json
import logging
import os
from collections import defaultdict
from functools import cache
from typing import Dict, List, Sequence, Set

from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import IPaperQuery, Paper, Variant

from .interfaces import IGetPapers

logger = logging.getLogger(__name__)


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
TRUTHSET_PAPER_KEYS = ["doi", "pmid", "pmcid", "paper_title", "link", "is_pmc_oa", "license"]
# These are the columns in the truthset that are specific to the variant.
TRUTHSET_VARIANT_KEYS = [
    "gene",
    "hgvs_c",
    "hgvs_p",
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
                logger.warning(f"Skipped {len(rows)} rows with no paper ID.")
                continue

            # For each paper, extract the paper-specific key/value pairs into a new dict.
            # These are repeated on every paper/variant row, so we can just take the first row.
            paper_data = {k: v for k, v in rows[0].items() if k in TRUTHSET_PAPER_KEYS}

            # Integrity checks.
            for row in rows:
                # Doublecheck if every row has the same values for the paper-specific keys.
                for key in TRUTHSET_PAPER_KEYS:
                    if paper_data[key] != row[key]:
                        logger.warning(f"Multiple values ({paper_data[key]} vs {row[key]}) for {key} ({paper_id}).")
                # Make sure the gene/variant columns are not empty.
                if not row["gene"] or not row["hgvs_p"]:
                    logger.warning(f"Missing gene or variant for {paper_id}.")

            # For each paper, extract the variant-specific key/value pairs into a new dict of dicts.
            variants = {Variant(r["gene"], r["hgvs_p"]): {k: r.get(k, "") for k in TRUTHSET_VARIANT_KEYS} for r in rows}
            # Create a Paper object with the extracted fields.
            papers.add(Paper(id=paper_id, evidence=variants, **paper_data))

        return papers

    def search(self, query: IPaperQuery) -> Set[Paper]:
        all_papers = self._load_truthset()
        query_genes = {v.gene for v in query.terms()}

        # Filter to just the papers with variant terms that have evidence for the genes specified in the query.
        return {p for p in all_papers if query_genes & {v.gene for v in p.evidence.keys()}}


class RemoteFileLibrary(IGetPapers):
    """A class for retrieving papers from PubMed."""

    def __init__(self, paper_client: IPaperLookupClient, max_papers: int = 5) -> None:
        """Initialize a new instance of the RemoteFileLibrary class.

        Args:
            paper_client (IPaperLookupClient): A class for searching and fetching papers.
            max_papers (int, optional): The maximum number of papers to retrieve. Defaults to 5.
        """
        self._paper_client = paper_client
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
        term = next(iter(query.terms())).gene
        paper_ids = self._paper_client.search(query=term, max_papers=self._max_papers)
        papers = {paper for paper_id in paper_ids if (paper := self._paper_client.fetch(paper_id)) is not None}
        return papers


class RareDiseaseFileLibrary(IGetPapers):
    # reimplement search from RemoteFileLibrary
    # paper_client.search
    # filer from that
    # return Set[Paper]
    # yaml - swap out at RemoteFileLibrary locations
    # test: does it return 0 of the papers we dont want
    """A class for filtering papers from PubMed."""

    def __init__(self, paper_client: IPaperLookupClient, max_papers: int = 5) -> None:
        """Initialize a new instance of the RemoteFileLibrary class.
        Args:
            paper_client (IPaperLookupClient): A class for searching and fetching papers.
            max_papers (int, optional): The maximum number of papers to retrieve. Defaults to 5.
        """
        self._paper_client = paper_client
        self._max_papers = max_papers

    def search(self, query: IPaperQuery):
        """Search for papers based on the given query.

        Args:
            query (IPaperQuery): The query to search for.

        Returns:
            Set[Paper]: The set of papers that match the query.
        """
        if len(query.terms()) > 1:
            raise NotImplementedError("Multiple term extraction not yet implemented.")

        # Get gene term
        term = next(iter(query.terms())).gene
        print(term)

        # Find paper IDs
        paper_ids = self._paper_client.search(query=term, max_papers=self._max_papers)

        # Extract the paper content that we care about (e.g. title, abstract, PMID, etc.)
        papers = {paper for paper_id in paper_ids if (paper := self._paper_client.fetch(paper_id)) is not None}

        # Call private function to filter for rare disease papers
        rare_disease_papers, non_rare_disease_papers, other_papers = self._filter_rare_disease_papers(papers)

        # Compare the ground truth papers PMIDs to the paper PMIDs that were found
        n_corr, n_miss, n_extra = self._compare_manual_ground_truth(term, rare_disease_papers)
        pn_corr, pn_miss, pn_extra = self._compare_pubmed_ground_truth(term, papers)

        return rare_disease_papers

    def _get_ground_truth_gene(self, gene: str):
        # Ground truth papers
        ground_truth_papers_pmids = {
            "EXOC2": ["32639540"],
            "RHOH": ["22850876"],
            "RNASEH1": ["30340744", "28508084", "31258551", "26094573", "33396418"],
            "DNAJC7": ["31768050", "32897108", "33193563", "34233860", "35039179", "35456894", "37870677", "38112783"],
            "PTCD3": ["36450274", "30706245", "30607703"],
            "ZNF423": ["32925911", "33531950", "33270637", "22863007"],
            "OTUD7A": ["33381903", "31997314"],
            "PRPH": ["15322088", "15446584", "20363051", "25299611", "26742954", "17045786", "30992453", "35714755"],
            "BAZ2B": [
                "31999386",
                "25363768",
                "28135719",
                "28554332",
                "28867142",
                "31398340",
                "31981491",
                "33057194",
                "37872713",
            ],
            "NDUFA2": ["18513682", "27159321", "28857146", "32154054"],
            "TOP2B": ["31198993", "31409799", "31953910", "35063500", "36450898", "37068767", "32128574"],
            "HYAL1": ["10339581", "21559944", "26322170"],
            "FOXE3": [
                "26854927",
                "26995144",
                "28418495",
                "29136273",
                "29314435",
                "29878917",
                "30078984",
                "31884615",
                "32224865",
                "32436650",
                "32499604",
                "32976546",
                "34046667",
                "35051625",
                "35170016",
                "36192130",
                "37628625",
                "37758467",
            ],
            "AHCY": [
                "19177456",
                "20852937",
                "22959829",
                "26095522",
                "26527160",
                "26974671",
                "28779239",
                "30121674",
                "31957987",
                "32689861",
                "33869213",
                "35789945",
                "38052822",
                "15024124",
                "16736098",
                "27848944",
                "35463910",
            ],
        }

        if gene in ground_truth_papers_pmids.keys():
            return ground_truth_papers_pmids[gene]
        else:
            return None

    # private function to compare the ground truth papers PMIDs to the papers that were found
    def _compare_manual_ground_truth(self, gene, r_d_papers) -> List[int]:
        """Compare the papers that were found to the ground truth papers.
        Args:
            paper (Paper): The paper to compare.
        Returns:
            number of correct papers (i.e. the number that match the ground truth)
            number of missed papers (i.e. the number that are in the ground truth but not in the papers that were found)
            number of extra papers (i.e. the number that are in the papers that were found but not in the ground truth)
        """
        n_correct = 0
        n_missed = 0
        n_extra = 0

        # Get all the PMIDs from all of the papers
        r_d_pmids = [paper.props.get("pmid", "Unknown") for paper in r_d_papers]
        # print("R_D_PMIDS ", r_d_pmids)

        ground_truth_papers_pmids = self._get_ground_truth_gene(gene)

        # Keep track of the correct and extra PMIDs to subtract from the ground truth papers PMIDs
        counted_pmids = []

        # For the gene, get the ground truth PMIDs from ground_truth_papers_pmids and compare the PMIDS to the PMIDS from the papers that were found
        # For any PMIDs that match, increment n_correct
        if ground_truth_papers_pmids is not None:
            for pmid in r_d_pmids:
                if pmid in ground_truth_papers_pmids:
                    n_correct += 1
                    counted_pmids.append(pmid)
                else:
                    n_extra += 1
                    counted_pmids.append(pmid)

            # For any PMIDs in the ground truth that are not in the papers that were found, increment n_missed, use counted_pmids to subtract from the ground truth papers PMIDs
            for pmid in ground_truth_papers_pmids:
                if pmid not in counted_pmids:
                    n_missed += 1

        else:
            n_correct = 0
            n_missed = 0
            n_extra = 0

        print(f"Tool against Manual Ground Truth - Correct: {n_correct}, Missed: {n_missed}, Extra: {n_extra}")
        return [n_correct, n_missed, n_extra]

    # private function to compare the ground truth papers PMIDs to the papers that were found
    def _compare_pubmed_ground_truth(self, gene, all_pubmed_papers) -> List[int]:
        """Compare the papers that were found to the ground truth papers.
        Args:
            paper (Paper): The paper to compare.
        Returns:
            number of correct papers (i.e. the number that match the ground truth)
            number of missed papers (i.e. the number that are in the ground truth but not in the papers that were found)
            number of extra papers (i.e. the number that are in the papers that were found but not in the ground truth)
        """
        n_correct = 0
        n_missed = 0
        n_extra = 0

        p_pmids = [paper.props.get("pmid", "Unknown") for paper in all_pubmed_papers]
        # print("P_PMIDS ", p_pmids)

        ground_truth_papers_pmids = self._get_ground_truth_gene(gene)

        # Keep track of the correct and extra PMIDs to subtract from the ground truth papers PMIDs
        counted_pmids = []

        # For the gene, get the ground truth PMIDs from ground_truth_papers_PMIDs and compare the PMIDS to the PMIDS from the papers that were found
        # For any PMIDs that match, increment n_correct
        if ground_truth_papers_pmids is not None:
            for pmid in p_pmids:
                if pmid in ground_truth_papers_pmids:
                    n_correct += 1
                    counted_pmids.append(pmid)
                else:
                    n_extra += 1
                    counted_pmids.append(pmid)

            # For any PMIDs in the ground truth that are not in the papers that were found, increment n_missed, use counted_pmids to subtract from the ground truth papers PMIDs
            for pmid in ground_truth_papers_pmids:
                if pmid not in counted_pmids:
                    n_missed += 1

        else:
            n_correct = 0
            n_missed = 0
            n_extra = 0

        print(f"PubMed against Manual Ground Truth - Correct: {n_correct}, Missed: {n_missed}, Extra: {n_extra}")
        return [n_correct, n_missed, n_extra]

    def _filter_rare_disease_papers(self, papers: Set[Paper]):
        """Filter papers to only include those that are related to rare diseases.
        Args:
            papers (Set[Paper]): The set of papers to filter.
        Returns:
            Set[Paper]: The set of papers that are related to rare diseases.
        """

        rare_disease_papers = set()
        non_rare_disease_papers = set()
        other_papers = set()

        for paper in papers:
            paper_title = paper.props.get("title", "Unknown")
            paper_abstract = paper.props.get("abstract", "Unknown")
            # print("paper_title", paper_title)

            if paper_title or paper_abstract is not None:

                # INCLUSION PRINCIPLES
                # Include papers that have these terms in the title
                if (
                    "variant"
                    or "variants"
                    or "rare disease"
                    or "rare variant"
                    or "rare variants"
                    or "monogenic"
                    or "monogenicity"
                    or "monoallelic"
                    or "syndromic"
                    or "inherited"
                    or "pathogenic"
                    or "benign"
                    or "inherited cancer" in paper_title.lower()
                ):
                    rare_disease_papers.add(paper)
                # If not in the title, check the abstract
                elif (
                    "variant"
                    or "variants"
                    or "rare disease"
                    or "rare variant"
                    or "rare variants"
                    or "monogenic"
                    or "monogenicity"
                    or "monoallelic"
                    or "syndromic"
                    or "inherited"
                    or "pathogenic"
                    or "benign"
                    or "inherited cancer" in paper_abstract.lower()
                ):
                    rare_disease_papers.add(paper)

                # EXCLUSION PRINCIPLES
                # Exclude papers that have these terms in the title.
                elif (
                    "digenic"
                    or "familial"
                    or "structural variant"
                    or "somatic"
                    or "somatic cancer"
                    or "cancer" in paper_title.lower()
                ):
                    non_rare_disease_papers.add(paper)
                # If not in the title, check the abstract
                elif (
                    "digenic"
                    or "familial"
                    or "structural variant"
                    or "somatic"
                    or "somatic cancer"
                    or "cancer" in paper_abstract.lower()
                ):
                    non_rare_disease_papers.add(paper)
                else:
                    other_papers.add(paper)

                # Exclude papers that are not written in English by scanning the title or abstract
                # TODO: Implement this

                # Exclude papers that only describe animal models and do not have human data
                # TODO: Implement this

        print("Rare Disease Papers: ", len(rare_disease_papers))
        print("Non-Rare Disease Papers: ", len(non_rare_disease_papers))
        print("Other Papers: ", len(other_papers))

        # Check if rare_disease_papers is empty or if non_rare_disease_papers is empty
        if len(rare_disease_papers) == 0:
            # print("No rare disease papers found.")
            rare_disease_papers = Set[Paper]
        if len(non_rare_disease_papers) == 0:
            # print("No non-rare disease papers found.")
            non_rare_disease_papers = Set[Paper]

        return rare_disease_papers, non_rare_disease_papers, other_papers
