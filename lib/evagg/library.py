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

    def __init__(
        self,
        paper_client: IPaperLookupClient,
        max_papers: int = 5,
        min_date: str = "1800/01/01",
        max_date: str = "2029/01/01",
        date_type: str = "pdat",
    ) -> None:
        """Initialize a new instance of the RemoteFileLibrary class.
        Args:
            paper_client (IPaperLookupClient): A class for searching and fetching papers.
            max_papers (int, optional): The maximum number of papers to retrieve. Defaults to 5.
        """
        self._paper_client = paper_client
        self._max_papers = max_papers
        self._min_date = min_date
        self._max_date = max_date
        self._date_type = date_type

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
        print("\nGENE: ", term, self._min_date)

        # Find paper IDs
        paper_ids = self._paper_client.search(
            query=term,
            max_papers=self._max_papers,  # ,
            # min_date=,
            # max_date=self._max_date,
            # date_type=self._date_type,
        )

        # Extract the paper content that we care about (e.g. title, abstract, PMID, etc.)
        papers = {paper for paper_id in paper_ids if (paper := self._paper_client.fetch(paper_id)) is not None}

        # Call private function to filter for rare disease papers
        rare_disease_papers, count_r_d_papers, non_rare_disease_papers, other_papers = self._filter_rare_disease_papers(
            papers
        )

        if count_r_d_papers != 0:
            # Compare the ground truth papers PMIDs to the paper PMIDs that were found
            # correct_pmids_papers, missed_pmids_papers, irrelevant_pmids_papers
            _, _, _ = self._compare_to_truth_or_tool(term, rare_disease_papers, 0)

        # pn_corr, pn_miss, pn_extra
        _, _, _ = self._compare_to_truth_or_tool(term, papers, 1)

        if count_r_d_papers == 0:
            rare_disease_papers = set()
        return rare_disease_papers

    def _get_ground_truth_gene(self, gene: str):
        # Ground truth papers
        ground_truth_papers_pmids = {
            "CTF1": ["11058912", "24503780", "26084686"],
            "FBN2": [
                "9714438",
                "9737771",
                "10797416",
                "11754102",
                "15121784",
                "17345643",
                "19006240",
                "19473076",
                "24585410",
                "24833718",
                "24899048",
                "27625873",
                "25834781",
                "25975422",
                "27007659",
                "27196565",
                "27606123",
                "27912749",
                "28379158",
                "28383543",
                "29742989",
                "29864108",
                "30147916",
                "30675029",
                "31316167",
                "31426022",
                "31506931",
                "32184806",
                "32900841",
                "33571691",
                "33638605",
                "34355836",
                "34456093",
                "35126451",
                "35360850",
                "35754816",
                "36800380",
                "36849876",
                "36936417",
                "37254066",
                "37399314",
                "37875969",
                "37962692",
                "38099230",
                "38114583",
                "38215673",
                "38326314",
            ],
            "EXOC2": ["32639540"],
            "RHOH": ["22850876"],
            "RNASEH1": ["30340744", "28508084", "31258551", "26094573", "33396418"],
            "TDO2": ["28285122", "35743796"],
            "DNAJC7": ["31768050", "32897108", "33193563", "34233860", "35039179", "35456894", "37870677", "38112783"],
            "PTCD3": ["36450274", "30706245", "30607703"],
            "ZNF423": ["32925911", "33531950", "33270637", "22863007"],
            "OTUD7A": ["33381903", "31997314"],
            "PRPH": ["15322088", "15446584", "20363051", "25299611", "26742954", "17045786", "30992453", "35714755"],
            "ADCY1": ["24482543"],
            "IGKC": ["32278584", "26853951", "4132042"],
            "SARS1": ["35790048", "36041817", "34570399"],
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
            "TAPBP": ["12149238"],
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
            "B4GAT1": ["34587870", "34602496", "25279699", "23877401"],
            "TOP2B": ["31198993", "31409799", "31953910", "35063500", "36450898", "37068767", "32128574"],
            "SLFN14": [
                "30431218",
                "31378119",
                "37140956",
                "36790527",
                "26280575",
                "33496736",
                "29678925",
                "37041648",
                "36237120",
                "26769223",
                "30536060",
            ],
            "HYAL1": ["10339581", "21559944", "26322170"],
            "RGS9": ["17826834", "14702087", "29107794", "19818506"],
            "MLH3": [
                "11586295",
                "12702580",
                "12800209",
                "15193445",
                "16885347",
                "16981255",
                "18521850",
                "19156873",
                "24549055",
                "25142776",
                "26149658",
                "26296701",
                "26542077",
                "27401157",
                "30573798",
                "30614234",
                "31297992",
                "32469048",
                "33117677",
                "33517345",
                "34008015",
                "34897210",
                "35475445",
                "37656691",
                "38201513",
                "28195393",
                "29212164",
            ],
            "COG4": [
                "19494034",
                "19651599",
                "21185756",
                "30290151",
                "31949312",
                "32078278",
                "32652690",
                "33340551",
                "33629572",
                "33688625",
                "34022244",
                "34298581",
                "34595172",
                "34603392",
                "35455576",
                "36393834",
            ],
            "JPH2": [
                "17476457",
                "17509612",
                "20694023",
                "22515980",
                "23973696",
                "24636942",
                "24935431",
                "25500949",
                "27471098",
                "30167555",
                "28393127",
                "30235249",
                "30384889",
                "31227780",
                "31478477",
                "32879264",
                "34394822",
                "34526680",
                "34861382",
                "35001666",
                "35238659",
                "36357925",
                "37371654",
                "29540472",
                "29165669",
            ],
            "NLGN3": [
                "12669065",
                "24126926",
                "25167861",
                "25363768",
                "25533962",
                "28263302",
                "15150161",
                "19360662",
                "20615874",
                "21808020",
                "15274046",
                "15679194",
                "16077734",
                "16429495",
                "16508939",
                "16648374",
                "18189281",
                "18361425",
                "18419324",
                "18555979",
                "19545994",
                "19645625",
                "20227402",
                "21569590",
                "22934180",
                "23020841",
                "23468870",
                "23761734",
                "23851596",
                "23871722",
                "24362370",
                "24570023",
                "25347860",
                "25464930",
                "25592157",
                "26469287",
                "27743928",
                "27782075",
                "28385162",
                "28584888",
                "28948087",
                "29028156",
                "30201312",
                "31119867",
                "31184401",
                "31827744",
                "32848696",
                "34262438",
                "34690695",
                "36280753",
                "36810932",
                "38255906",
            ],
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
            "GRXCR2": ["24619944", "28383030", "32048449", "30157177", "33528103"],
            "EMC1": [
                "36799557",
                "35234901",
                "37187958",
                "26942288",
                "32092440",
                "29271071",
                "38161285",
                "35684946",
                "37554197",
                "32869858",
            ],
            "PEX11G": ["26935981"],
            "KMO": ["23459468"],
            "MIB1": [],
            "MPST": [],
            "SLC38A9": [],
            "HYKK": [],
            "CPT1B": [],
            "TNNC2": [],
            "NPPA": [],
            "LRRC10": ["26017719", "29431102", "28032242", "27536250", "29431105", "31270560"],
            "TOPBP1": ["24702692", "34199176"],
            "PEX11A": [],
            "DECR1": [],
            "ACAT2": [],
            "KIF1B": [
                "11389829",
                "30126838",
                "16163269",
                "16877806",
                "18726616",
                "22595495",
                "25802885",
                "27986441",
                "32298515",
                "33112832",
                "33362715",
                "33397043",
                "34169998",
                "35046208",
                "37564981",
                "37780619",
            ],
        }

        if gene in ground_truth_papers_pmids.keys():
            return ground_truth_papers_pmids[gene]
        else:
            return None

    # private function to compare the ground truth papers PMIDs to the papers that were found
    def _compare_to_truth_or_tool(self, gene, r_d_papers, pubmed):
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
        n_irrelevant = 0

        # Get all the PMIDs from all of the papers
        r_d_pmids = [(paper.props.get("pmid", "Unknown"), paper.props.get("title", "Unknown")) for paper in r_d_papers]

        ground_truth_papers_pmids = self._get_ground_truth_gene(gene)

        # Keep track of the correct and extra PMIDs to subtract from the ground truth papers PMIDs
        counted_pmids = []
        correct_pmids_papers = []
        missed_pmids_papers = []
        irrelevant_pmids_papers = []

        # For the gene, get the ground truth PMIDs from ground_truth_papers_pmids and compare the PMIDS to the PMIDS from the papers that were found
        # For any PMIDs that match, increment n_correct
        if ground_truth_papers_pmids is not None:
            for pmid, title in r_d_pmids:
                if pmid in ground_truth_papers_pmids:
                    n_correct += 1
                    counted_pmids.append(pmid)
                    correct_pmids_papers.append((pmid, title))

                else:
                    n_irrelevant += 1
                    counted_pmids.append(pmid)
                    irrelevant_pmids_papers.append((pmid, title))

            # For any PMIDs in the ground truth that are not in the papers that were found, increment n_missed, use counted_pmids to subtract from the ground truth papers PMIDs
            for pmid in ground_truth_papers_pmids:
                if pmid not in counted_pmids:
                    missed_paper_title = self._paper_client.fetch(str(pmid))
                    missed_paper_title = (
                        missed_paper_title.props.get("title", "Unknown")
                        if missed_paper_title is not None
                        else "Unknown"
                    )
                    # print("missed: ", pmid, missed_paper_title)
                    n_missed += 1
                    missed_pmids_papers.append((pmid, missed_paper_title))

        else:
            n_correct = (0, "NA")
            n_missed = (0, "NA")
            n_irrelevant = (0, "NA")

        # todo: filter r_d_papers to only include those that overlap with correct_pmids, missed_pmids, and extra_pmids so that I can use the props to print the titles.
        if pubmed:  # comparing tool to PubMed
            print("\nOf PubMed papers...")
            print("Pubmed # Correct Papers: ", len(correct_pmids_papers))
            print("Pubmed # Missed Papers: ", len(missed_pmids_papers))
            print("Pubmed # Irrelevant Papers: ", len(irrelevant_pmids_papers))

            print(f"\nFound Pubmed {len(correct_pmids_papers)} correct.")
            for i, p in enumerate(correct_pmids_papers):
                print("*", i + 1, "*", p[0], "*", p[1])

            print(f"\nFound Pubmed {len(missed_pmids_papers)} missed.")
            for i, p in enumerate(missed_pmids_papers):
                print("*", i + 1, "*", p[0], "*", p[1])

            print(f"\nFound Pubmed {len(irrelevant_pmids_papers)} irrelevant.")
            for i, p in enumerate(irrelevant_pmids_papers):
                print("*", i + 1, "*", p[0], "*", p[1])
        else:  # Comparing tool to manual ground truth data
            print("\nOf the rare disease papers...")
            print("Tool # Correct Papers: ", len(correct_pmids_papers))
            print("Tool # Missed Papers: ", len(missed_pmids_papers))
            print("Tool # Irrelevant Papers: ", len(irrelevant_pmids_papers))

            print(f"\nFound tool {len(correct_pmids_papers)} correct.")
            for i, p in enumerate(correct_pmids_papers):
                print("*", i + 1, "*", p[0], "*", p[1])

            print(f"\nFound tool {len(missed_pmids_papers)} missed.")
            for i, p in enumerate(missed_pmids_papers):
                print("*", i + 1, "*", p[0], "*", p[1])

            print(f"\nFound tool {len(irrelevant_pmids_papers)} irrelevant.")
            for i, p in enumerate(irrelevant_pmids_papers):
                print("*", i + 1, "*", p[0], "*", p[1])

        return correct_pmids_papers, missed_pmids_papers, irrelevant_pmids_papers

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
                "rare disease",
                "rare variant",
                "disorder",
                "syndrome",
                "-emia",
                "-cardia",
                "-phagia",
                "pathogenic",
                "benign",
                "inherited cancer",
                "germline",
            ]

            inclusion_keywords = inclusion_keywords + [word + "s" for word in inclusion_keywords]

            inclusion_keywords_odd_plurals = [
                "-lepsy",
                "-lepsies",
                "-pathy",
                "-pathies",
                "-osis",
                "-oses",
                "variant of unknown significance",
                "variants of unknown significance",
                "variant of uncertain significance" "variants of uncertain significance",
            ]

            inclusion_keywords_no_plural = [
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
                "disease causing",
            ]

            inclusion_keywords = inclusion_keywords + inclusion_keywords_odd_plurals + inclusion_keywords_no_plural

            exclusion_keywords = [
                "digenic",
                "familial",
                "structural variant",
                "structural variants",
                "somatic",
                "somatic cancer",
                "somatic cancers",
                "cancer",
                "cancers",
                "CNV",
                "CNVs",
                "copy number variant",
                "copy number variants",
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

        print("Rare Disease Papers: ", len(rare_disease_papers))
        print("Non-Rare Disease Papers: ", len(non_rare_disease_papers))
        print("Other Papers: ", len(other_papers))

        # Check if rare_disease_papers is empty or if non_rare_disease_papers is empty
        cnt_r_d_p = 1
        if len(rare_disease_papers) == 0:
            # print("No rare disease papers found.")
            cnt_r_d_p = 0
            rare_disease_papers = Set[Paper]
        if len(non_rare_disease_papers) == 0:
            # print("No non-rare disease papers found.")
            non_rare_disease_papers = Set[Paper]

        return rare_disease_papers, cnt_r_d_p, non_rare_disease_papers, other_papers
