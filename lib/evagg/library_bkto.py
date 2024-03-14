import csv
import json
import logging
import os
from collections import defaultdict
from functools import cache
from typing import Any, Dict, List, Sequence, Set

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

    def get_papers(self, query: Dict[str, Any]) -> Set[Paper]:
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

            # For each paper, irrelevantct the paper-specific key/value pairs into a new dict.
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

    def get_papers(self, query: Dict[str, Any]) -> Set[Paper]:
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

    def get_papers(self, query: Dict[str, Any]) -> Set[Paper]:
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
    """A class for filtering papers from PubMed."""

    def __init__(self, paper_client: IPaperLookupClient, max_papers: int = 5) -> None:
        """Initialize a new instance of the RemoteFileLibrary class.
        Args:
            paper_client (IPaperLookupClient): A class for searching and fetching papers.
            max_papers (int, optional): The maximum number of papers to retrieve. Defaults to 5.
        """
        self._paper_client = paper_client
        self._max_papers = max_papers

    def get_papers(self, query: Dict[str, Any]):
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
        logger.info(f"\nSearching for papers related to {term}.")

        # Find paper IDs
        paper_ids = self._paper_client.search(
            query=term,
            max_papers=self._max_papers,
            mindate=query.something,
        )

        # Extract the paper content that we care about (e.g. title, abstract, PMID, etc.)
        papers = {paper for paper_id in paper_ids if (paper := self._paper_client.fetch(paper_id)) is not None}

        print("\n\n GENE: ", term, "*")
        # Call private function to filter for rare disease papers
        rd_papers, non_rd_papers, irr_papers = self._filter_rare_disease_papers(papers)

        if "id:" in str(rd_papers):
            print(f"\nFound {len(rd_papers)} rare disease papers.")
            for i, p in enumerate(rd_papers):
                print("*", i + 1, "*", p.props["pmid"], "*", p.props["title"])

        if "id:" in str(non_rd_papers):
            print(f"\nFound {len(non_rd_papers)} NON-rare disease papers.")
            for i, p in enumerate(non_rd_papers):
                print("*", i + 1, "*", p.props["pmid"], "*", p.props["title"])

        if "id:" in str(irr_papers):
            print(f"\nFound {len(irr_papers)} irrelevant rare disease papers.")
            for i, p in enumerate(irr_papers):
                print("*", i + 1, "*", p.props["pmid"], "*", p.props["title"])

        # rare_disease_papers = self._filter_rare_disease_papers(papers)[
        #    0
        # ]  # chose not to return dummy variables and to use indexing instead as more memory efficient

        return rd_papers

    def split_papers_into_categories(self, query: IPaperQuery):
        if len(query.terms()) > 1:
            raise NotImplementedError("Multiple term extraction not yet implemented.")

        # Get gene term
        term = next(iter(query.terms())).gene
        print(f"\nSearching for papers related to {term}.")

        # Find paper IDs
        print("term", term)
        paper_ids = self._paper_client.search(query=term, max_papers=self._max_papers)

        # Extract the paper content that we care about (e.g. title, abstract, PMID, etc.)
        papers = {paper for paper_id in paper_ids if (paper := self._paper_client.fetch(paper_id)) is not None}

        # Call private function to filter for rare disease papers
        rare_disease_papers, non_rare_disease_papers, other_papers = self._filter_rare_disease_papers(papers)
        return rare_disease_papers, non_rare_disease_papers, other_papers

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

            inclusion_keywords = [
                "variant",
                "variants",
                "rare disease",
                "rare variant",
                "rare variants",
                "disorder",
                "syndrome",
                "mendelian",
                "monogenic",
                "monogenicity",
                "monoallelic",
                "syndromic",
                "inherited",
                "hereditary",
                "dominant",
                "recessive",
                "de novo",
                "VUS",
                "variant of unknown significance",
                "variant of uncertain significance",
                "pathogenic",
                "benign",
                "inherited cancer",
            ]

            exclusion_keywords = [
                "digenic",
                "structural variant",
                "somatic",
                "somatic cancer",
                "cancer",
                "CNV",
                "copy number variant",
            ]
            # include
            if paper_title is not None and any(keyword in paper_title.lower() for keyword in inclusion_keywords):
                rare_disease_papers.add(paper)
            elif paper_abstract is not None and any(
                keyword in paper_abstract.lower() for keyword in inclusion_keywords
            ):
                rare_disease_papers.add(paper)
            # exclude
            elif paper_title is not None and any(keyword in paper_title.lower() for keyword in exclusion_keywords):
                non_rare_disease_papers.add(paper)
            elif paper_abstract is not None and any(
                keyword in paper_abstract.lower() for keyword in exclusion_keywords
            ):
                non_rare_disease_papers.add(paper)
            # other
            else:
                other_papers.add(paper)

            # Exclude papers that are not written in English by scanning the title or abstract
            # TODO: Implement this

            # Exclude papers that only describe animal models and do not have human data
            # TODO: Implement this

        print(f"Found {len(rare_disease_papers)} rare disease papers.")
        print(f"Found {len(non_rare_disease_papers)} non-rare disease papers.")
        print(f"Found {len(other_papers)} other papers.")

        # logger.info(f"Found {len(rare_disease_papers)} rare disease papers.")
        # logger.info(f"Found {len(non_rare_disease_papers)} non-rare disease papers.")
        # logger.info(f"Found {len(other_papers)} other papers.")

        # Check if rare_disease_papers is empty or if non_rare_disease_papers is empty
        if len(rare_disease_papers) == 0:
            # print("No rare disease papers found.")
            rare_disease_papers = Set[Paper]
        if len(non_rare_disease_papers) == 0:
            # print("No non-rare disease papers found.")
            non_rare_disease_papers = Set[Paper]

        return rare_disease_papers, non_rare_disease_papers, other_papers

    # Run through paper finding benchmarks on the training set
    def run_paper_finding_benchmarks(term, papers, rare_disease_papers, get_ground_truth_pmids):

        # Calculate # correct, # missed, and # irrelevant papers for our tool against the manual ground truth
        ea_correct_pmids, ea_missed_pmids, ea_irrelevant_pmids = calc_metrics_against_manual_ground_truth(
            term, rare_disease_papers, get_ground_truth_pmids
        )

        # if not rare_disease_papers:
        #     rare_disease_papers = set()

        # Calculate # correct, # missed, and # irrelevant papers for our tool against PubMed
        pub_correct_pmids, pub_missed_pmids, pub_irrelevant_pmids = calc_metrics_against_manual_ground_truth(
            term, papers, get_ground_truth_pmids
        )

        return (
            ea_correct_pmids,
            ea_missed_pmids,
            ea_irrelevant_pmids,
            pub_correct_pmids,
            pub_missed_pmids,
            pub_irrelevant_pmids,
        )

    # Get the ground truth PMIDs for a gene.
    def get_ground_truth_pmids(gene, mgt_gene_pmids_dict):
        if gene in mgt_gene_pmids_dict.keys():
            return mgt_gene_pmids_dict[gene]
        else:
            return None

    # Compare the ground truth papers PMIDs to the papers (our tool or PubMed's papers) that were found
    def calc_metrics_against_manual_ground_truth(term, input_papers, mgt_gene_pmids_dict):
        """Compare the papers that were found to the ground truth papers.
        Args:
            paper (Paper): The paper to compare.
        Returns:
            number of correct papers (i.e. the number that match the ground truth)
            number of missed papers (i.e. the number that are in the ground truth but not in the papers that were found)
            number of irrelevant papers (i.e. the number that are in the papers that were found but not in the ground truth)
        """

        # Get all the PMIDs from all of the papers
        input_paper_pmids = [paper.props.get("pmid", "Unknown") for paper in input_papers]

        # Get the ground truth PMIDs list for the gene
        ground_truth_gene_pmids = get_ground_truth_pmids(term, mgt_gene_pmids_dict)

        # Keep track of the correct, missed, and irrelevant PMIDs to subtract from the ground truth papers PMIDs
        counted_pmids = []
        correct_pmids = []
        missed_pmids = []
        irrelevant_pmids = []

        # For each gene, get ground truth PMIDs from ground_truth_papers_pmids and compare those PMIDS to the PMIDS from the papers that the tool found (e.g. our ev. agg. tool or PubMed).
        if ground_truth_gene_pmids is not None:
            for pmid in input_paper_pmids:
                if pmid in ground_truth_gene_pmids:
                    counted_pmids.append(pmid)
                    correct_pmids.append(pmid)

                else:
                    print("irrelevant pmid: ", pmid)
                    counted_pmids.append(pmid)
                    irrelevant_pmids.append(pmid)

            # For any PMIDs in the ground truth that are not in the papers that were found, increment n_missed, use counted_pmids to subtract from the ground truth papers PMIDs
            for pmid in ground_truth_gene_pmids:
                if pmid not in counted_pmids:
                    print("missed pmid: ", pmid)
                    missed_pmids.append(pmid)

        print(
            f"Compared to Manual Ground Truth - Correct: {len(correct_pmids)}, Missed: {len(missed_pmids)}, Irrelevant: {len(irrelevant_pmids)}"
        )

        return correct_pmids, missed_pmids, irrelevant_pmids

    def get_paper_titles(input_pmids, input_papers):
        # Filter input_papers and create 3 objects that each have all of the paper titles, which correspond to the pmids in correct_pmids, missed_pmids, and irrelevant_pmids.
        input_pmid_titles = [
            paper.props.get("title", "Unknown")
            for paper in input_papers
            if paper.props.get("pmid", "Unknown") in input_pmids
        ]
        # paper_abstract = paper.props.get("abstract", "Unknown")
        return input_pmid_titles