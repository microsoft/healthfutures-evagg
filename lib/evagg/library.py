import asyncio
import csv
import json
import logging
import os
import re
from collections import defaultdict
from datetime import date
from functools import cache
from typing import Any, Dict, List, Sequence, Set, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from lib.evagg.llm import IPromptClient
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import ICreateVariants, Paper

from .disease_keywords import EXCLUSION_KEYWORDS, INCLUSION_KEYWORDS
from .interfaces import IGetPapers

logger = logging.getLogger(__name__)
# TODO: ways to improve:
#       - process full text of paper to filter to rare disease papers when PMC OA


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

    def get_papers(self, query: Dict[str, Any]) -> Sequence[Paper]:
        # Dummy implementation that returns all papers regardless of query.
        all_papers = list(self._load().values())
        return all_papers


# These are the columns in the truthset that are specific to the paper.
TRUTHSET_PAPER_KEYS = ["paper_id", "pmid", "pmcid", "paper_title", "license", "link"]
TRUTHSET_PAPER_KEYS_MAPPING = {"paper_id": "id", "paper_title": "title"}
# These are the columns in the truthset that are specific to the evidence.
TRUTHSET_EVIDENCE_KEYS = [
    "gene",
    "paper_variant",
    "hgvs_c",
    "hgvs_p",
    "transcript",
    "individual_id",
    "phenotype",
    "zygosity",
    "variant_inheritance",
    "variant_type",
    "functional_study",
    # "gnomad_frequency",
    "study_type",
    "notes",
]


class TruthsetFileLibrary(IGetPapers):
    """A class for retrieving papers from a truthset file."""

    _variant_factory: ICreateVariants
    _paper_client: IPaperLookupClient

    def __init__(self, file_path: str, variant_factory: ICreateVariants, paper_client: IPaperLookupClient) -> None:
        self._file_path = file_path
        self._variant_factory = variant_factory
        self._paper_client = paper_client

    def _process_paper(self, paper_id: str, rows: List[Dict[str, str]]) -> Paper:
        """Process a paper from the truthset file and return a Paper object with associated evidence."""
        logger.info(f"Processing {len(rows)} variants/patients for {paper_id}.")

        # Fetch a Paper object with the extracted fields based on the PMID.
        assert paper_id.startswith("pmid:"), f"Paper ID {paper_id} does not start with 'pmid:'."
        if not (paper := self._paper_client.fetch(paper_id[len("pmid:") :], include_fulltext=True)):
            raise ValueError(f"Failed to fetch paper with ID {paper_id}.")

        for row in rows:
            # Doublecheck the truthset row has the same values as the Paper for all paper-specific keys.
            for row_key in TRUTHSET_PAPER_KEYS:
                key = TRUTHSET_PAPER_KEYS_MAPPING.get(row_key, row_key)
                if paper.props[key] != row[row_key]:
                    raise ValueError(f"Truthset mismatch for {paper.id} {key}: {paper.props[key]} vs {row[row_key]}.")

            # Parse the variant from the HGVS c. or p. description.
            text_desc = row["hgvs_c"] if row["hgvs_c"].startswith("c.") else row["hgvs_p"]
            variant = self._variant_factory.parse(text_desc, row["gene"], row["transcript"])

            # Create an evidence dictionary from the variant/patient-specific columns.
            evidence = {key: row.get(key, "") for key in TRUTHSET_EVIDENCE_KEYS}
            # Add a unique identifier for this combination of paper, variant, and individual ID.
            evidence["pub_ev_id"] = f"{paper.id}:{variant.hgvs_desc}:{row['individual_id']}".replace(" ", "")
            paper.evidence[(variant, row["individual_id"])] = evidence

        return paper

    @cache
    def _load_truthset(self) -> Set[Paper]:
        row_count = 0
        # Group the rows by paper ID.
        paper_groups = defaultdict(list)
        with open(self._file_path) as tsvfile:
            header = [h.strip() for h in tsvfile.readline().split("\t")]
            reader = csv.reader(tsvfile, delimiter="\t")
            for line in reader:
                fields = dict(zip(header, [field.strip() for field in line]))
                paper_groups[fields["paper_id"]].append(fields)
                row_count += 1

        logger.info(f"Loaded {row_count} rows with {len(paper_groups)} papers from {self._file_path}.")
        # Process each paper row group into a Paper object with truthset evidence filled in.
        papers = {self._process_paper(paper_id, rows) for paper_id, rows in paper_groups.items()}
        # Make sure that each evidence truthset row is unique across the truthset.
        assert len({ev["pub_ev_id"] for p in papers for ev in p.evidence.values()}) == row_count
        return papers

    def get_papers(self, query: Dict[str, Any]) -> Sequence[Paper]:
        """For the TruthsetFileLibrary, query is expected to be a gene symbol."""
        all_papers = self._load_truthset()

        if gene_symbol := query.get("gene_symbol"):
            # Filter to just the papers with variants that have evidence for the gene specified in the query.
            return [p for p in all_papers if gene_symbol in {v[0].gene_symbol for v in p.evidence.keys()}]
        return []


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

    def get_papers(self, query: Dict[str, Any]) -> Sequence[Paper]:
        """Search for papers based on the given query.

        Args:
            query (IPaperQuery): The query to search for.

        Returns:
            Sequence[Paper]: The set of papers that match the query.
        """
        # Get gene term
        if not query.get("gene_symbol"):
            raise ValueError("Minimum requirement to search is to input a gene symbol.")

        paper_ids = self._paper_client.search(query=query["gene_symbol"])
        papers = [paper for paper_id in paper_ids if (paper := self._paper_client.fetch(paper_id)) is not None]
        return papers


class RareDiseaseFileLibrary(IGetPapers):
    """A class for fetching and categorizing disease papers from PubMed."""

    CATEGORIES = ["rare disease", "other"]

    def __init__(
        self,
        paper_client: IPaperLookupClient,
        llm_client: IPromptClient,
        allowed_categories: Sequence[str] | None = None,
    ) -> None:
        """Initialize a new instance of the RareDiseaseFileLibrary class.

        Args:
            paper_client (IPaperLookupClient): A class for searching and fetching papers.
            llm_client (IPromptClient): A class to leverage LLMs to filter to the right papers.
            allowed_categories (Sequence[str], optional): The categories of papers to allow. Defaults to "rare disease".
        """
        # TODO: go back and incorporate the idea of paper_types that can be passed into RareDiseaseFileLibrary,
        # so that the user of this class can specify which types of papers they want to filter for.
        self._paper_client = paper_client
        self._llm_client = llm_client
        self._allowed_categories = allowed_categories if allowed_categories is not None else ["rare disease"]

    def _get_keyword_category(self, paper: Paper) -> str:
        """Categorize papers based on keywords in the title and abstract."""
        title = paper.props.get("title") or ""
        abstract = paper.props.get("abstract") or ""

        # TODO: Exclude papers that are not written in English by scanning the title or abstract
        # TODO: Exclude papers that only describe animal models and do not have human data

        def _has_keywords(text: str, keywords: List[str]) -> bool:
            """Check if a text contains any of the keywords."""
            if not text:
                return False

            full_text = text.lower()
            split_text = full_text.split()

            for keyword in keywords:
                if keyword.startswith("-"):
                    if any(re.search(f"{keyword[1:]}$", word) for word in split_text):
                        return True
                else:
                    if keyword in full_text:
                        return True
            return False

        # Check if the paper should be included in the rare disease category.
        if _has_keywords(title, INCLUSION_KEYWORDS) or _has_keywords(abstract, INCLUSION_KEYWORDS):
            return "rare disease"
        # Check if the paper should be included in the non-rare disease category.
        if _has_keywords(title, EXCLUSION_KEYWORDS) or _has_keywords(abstract, EXCLUSION_KEYWORDS):
            return "non-rare disease"
        # If the paper doesn't fit in the other categories, add it to the other category.
        return "other"

    def _get_paper_texts(self, paper: Paper) -> Dict[str, Any]:
        texts: Dict[str, Any] = {}
        texts["full_text"] = "\n".join(paper.props.get("full_text_sections", []))
        texts["tables"] = "\n".join(self._get_tables_from_paper(paper).values())
        return texts

    def _get_tables_from_paper(self, paper: Paper) -> Dict[str, str]:
        tables: Dict[str, str] = {}
        root = paper.props.get("full_text_xml")
        if root is not None:
            for passage in root.findall("./passage"):
                if bool(passage.findall("infon[@key='section_type'][.='TABLE']")):
                    id = passage.find("infon[@key='id']").text
                    if not id:
                        logger.error("No id for table, using None as key")
                    tables[id] = tables.get(id, "") + "\n" + passage.find("text").text
        return tables

    async def _get_llm_category(self, paper: Paper) -> str:
        paper_finding_txt = (
            "paper_finding.txt" if paper.props.get("full_text_xml") is None else "paper_finding_full_text.txt"
        )
        parameters = (
            self._get_paper_texts(paper)
            if paper_finding_txt == "paper_finding_full_text.txt"
            else {
                "abstract": paper.props.get("abstract") or "no abstract",
                "title": paper.props.get("title") or "no title",
            }
        )
        response = await self._llm_client.prompt_file(
            user_prompt_file=os.path.join(os.path.dirname(__file__), "content", "prompts", paper_finding_txt),
            system_prompt="Extract field",
            params=parameters,
            prompt_settings={"prompt_tag": "paper_category", "temperature": 0.8},
        )

        if isinstance(response, str):
            result = response
        else:
            logger.warning(f"LLM failed to return a valid categorization response for {paper.id}: {response}")

        if result in self.CATEGORIES:
            return result

        return "other"

    async def _apply_chain_of_thought(self, paper: Paper) -> str:
        """Categorize papers based on LLM prompts."""
        # Check if the paper is full text or not
        # if paper.props.get("full_text_xml") is not None:
        #     parameters = self._get_paper_texts(paper)
        # else:
        parameters = {
            "abstract": paper.props.get("abstract") or "no abstract",
            "title": paper.props.get("title") or "no title",
        }

        # flexible
        process_response = await self._llm_client.prompt_file(
            user_prompt_file=os.path.join(
                os.path.dirname(__file__), "content", "prompts", "paper_finding_directions.txt"
            ),
            system_prompt="Extract field",
            params=parameters,
            prompt_settings={"prompt_tag": "paper_category", "temperature": 0.8},
        )

        # Append output requirements to process_response before saving
        # if paper.props.get("full_text_xml") is not None:
        #     phrases = (
        #         "\n It is essential that you provide your response as a single string: "
        #         "\"rare disease\" or \"other\" based on your classification. "
        #         "The only valid values in your output response should be \"rare disease\" or \"other\".\n\n"
        #         f"Below is the full text of the paper, which includes the title, abstract, full paper, and captions:\n\n"
        #         f"Full text: {parameters['full_text']}\n"
        #     )
        # else:
        phrases = (
            "\n It is essential that you provide your response as a single string: "
            "\"rare disease\" or \"other\" based on your classification. "
            "The only valid values in your output response should be \"rare disease\" or \"other\".\n\n"
            f"Below are the title and abstract:\n\n"
            f"Title: {paper.props.get('title') or 'no title'}\n"
            f"Abstract: {paper.props.get('abstract') or 'no abstract'}\n"
            
            "Below are few shot examples of papers that are classified as rare disease:\n"
            "Gene: SLFN14, PMID: 29678925\n"
            "Title: Role of the novel endoribonuclease SLFN14 and its disease-causing mutations in ribosomal degradation.\n"
            "Abstract: Platelets are anucleate and mostly ribosome-free cells within the bloodstream, derived from megakaryocytes within bone marrow and crucial for cessation of bleeding at sites of injury. Inherited thrombocytopenias are a group of disorders characterized by a low platelet count and are frequently associated with excessive bleeding. SLFN14 is one of the most recently discovered genes linked to inherited thrombocytopenia where several heterozygous missense mutations in SLFN14 were identified to cause defective megakaryocyte maturation and platelet dysfunction. Yet, SLFN14 was recently described as a ribosome-associated protein resulting in rRNA and ribosome-bound mRNA degradation in rabbit reticulocytes. To unveil the cellular function of SLFN14 and the link between SLFN14 and thrombocytopenia, we examined SLFN14 (WT/mutants) in in vitro models. Here, we show that all SLFN14 variants colocalize with ribosomes and mediate rRNA endonucleolytic degradation. Compared to SLFN14 WT, expression of mutants is dramatically reduced as a result of post-translational degradation due to partial misfolding of the protein. Moreover, all SLFN14 variants tend to form oligomers. These findings could explain the dominant negative effect of heterozygous mutation on SLFN14 expression in patients' platelets. Overall, we suggest that SLFN14 could be involved in ribosome degradation during platelet formation and maturation.\n"

            "Gene: SARS1, PMID: 35790048\n"
            "Title: WARS1 and SARS1: Two tRNA synthetases implicated in autosomal recessive microcephaly.\n"
            "Abstract: Aminoacylation of transfer RNA (tRNA) is a key step in protein biosynthesis, carried out by highly specific aminoacyl-tRNA synthetases (ARSs). ARSs have been implicated in autosomal dominant and autosomal recessive human disorders. Autosomal dominant variants in tryptophanyl-tRNA synthetase 1 (WARS1) are known to cause distal hereditary motor neuropathy and Charcot-Marie-Tooth disease, but a recessively inherited phenotype is yet to be clearly defined. Seryl-tRNA synthetase 1 (SARS1) has rarely been implicated in an autosomal recessive developmental disorder. Here, we report five individuals with biallelic missense variants in WARS1 or SARS1, who presented with an overlapping phenotype of microcephaly, developmental delay, intellectual disability, and brain anomalies. Structural mapping showed that the SARS1 variant is located directly within the enzyme's active site, most likely diminishing activity, while the WARS1 variant is located in the N-terminal domain. We further characterize the identified WARS1 variant by showing that it negatively impacts protein abundance and is unable to rescue the phenotype of a CRISPR/Cas9 wars1 knockout zebrafish model. In summary, we describe two overlapping autosomal recessive syndromes caused by variants in WARS1 and SARS1, present functional insights into the pathogenesis of the WARS1-related syndrome and define an emerging disease spectrum: ARS-related developmental disorders with or without microcephaly.\n"

            "Gene: MLH3, PMID: 26296701\n"
            "Title: Exome sequencing reveals frequent deleterious germline variants in cancer susceptibility genes in women with invasive breast cancer undergoing neoadjuvant chemotherapy.\n"
            "Abstract: When sequencing blood and tumor samples to identify targetable somatic variants for cancer therapy, clinically relevant germline variants may be uncovered. We evaluated the prevalence of deleterious germline variants in cancer susceptibility genes in women with breast cancer referred for neoadjuvant chemotherapy and returned clinically actionable results to patients. Exome sequencing was performed on blood samples from women with invasive breast cancer referred for neoadjuvant chemotherapy. Germline variants within 142 hereditary cancer susceptibility genes were filtered and reviewed for pathogenicity. Return of results was offered to patients with deleterious variants in actionable genes if they were not aware of their result through clinical testing. 124 patients were enrolled (median age 51) with the following subtypes: triple negative (n = 43, 34.7%), HER2+ (n = 37, 29.8%), luminal B (n = 31, 25%), and luminal A (n = 13, 10.5%). Twenty-eight deleterious variants were identified in 26/124 (21.0%) patients in the following genes: ATM (n = 3), BLM (n = 1), BRCA1 (n = 4), BRCA2 (n = 8), CHEK2 (n = 2), FANCA (n = 1), FANCI (n = 1), FANCL (n = 1), FANCM (n = 1), FH (n = 1), MLH3 (n = 1), MUTYH (n = 2), PALB2 (n = 1), and WRN (n = 1). 121/124 (97.6%) patients consented to return of research results. Thirteen (10.5%) had actionable variants, including four that were returned to patients and led to changes in medical management. Deleterious variants in cancer susceptibility genes are highly prevalent in patients with invasive breast cancer referred for neoadjuvant chemotherapy undergoing exome sequencing. Detection of these variants impacts medical management.\n"

            "Gene: FBN2, PMID: 9714438\n"
            "Title: Clustering of FBN2 mutations in patients with congenital contractural arachnodactyly indicates an important role of the domains encoded by exons 24 through 34 during human development.\n"
            "Abstract: Congenital contractural arachnodactyly (CCA) is an autosomal dominant condition phenotypically related to Marfan syndrome (MFS). CCA is caused by mutations in FBN2, whereas MFS results from mutations in FBN1. FBN2 mRNA extracted from 12 unrelated CCA patient cell strains was screened for mutations, and FBN2 mutations were identified in six of these samples. All of the identified FBN2 mutations cluster in a limited region of the gene, a region where mutations in FBN1 produce the severe, congenital form of MFS (so-called neonatal MFS). Furthermore, three of the identified mutations occur in the FBN2 locations exactly corresponding to FBN1 mutations that have been reported in cases of neonatal MFS. These mutations indicate that this central region of both of the fibrillins plays a critical role in human embryogenesis. The limited region of FBN2 that can be mutated to cause CCA may also help to explain the rarity of CCA compared to MFS.\n"
        )

        process_response = str(process_response) + phrases

        # Write the output of the variable to a file
        with open(
            os.path.join(
                os.path.dirname(__file__), "content", "prompts",
                f"paper_finding_process_{paper.id.replace('pmid:', '')}.txt"
            ),
            "w"
        ) as f:
            f.write(str(process_response))
            # TODO: Saving the paper finding process per paper is useful for benchmarking, not necessary for the
            # final product. Should I not save this to .out and instead just override w/ each new paper?

        # flexible
        classification_response = await self._llm_client.prompt_file(
            user_prompt_file=os.path.join(
                os.path.dirname(__file__), "content", "prompts", f"paper_finding_process_{paper.id.replace("pmid:", "")}.txt"
            ),
            system_prompt="Extract field",
            params={
                "abstract": paper.props.get("abstract") or "no abstract",
                "title": paper.props.get("title") or "no title",
            },
            prompt_settings={"prompt_tag": "paper_category", "temperature": 0.8},
        )
        if isinstance(classification_response, str):
            result = classification_response
        else:
            logger.warning(
                f"LLM failed to return a valid categorization response for {paper.id}: {classification_response}"
            )

        if result in self.CATEGORIES:
            return result

        return "other"

    def get_llm_category_w_few_shot(self, paper: Paper) -> str:
        paper_finding_txt = (
            "paper_finding.txt" if paper.props.get("full_text_xml") is None else "paper_finding_full_text.txt"
        )
        parameters = (
            self._get_paper_texts(paper)
            if paper_finding_txt == "paper_finding_full_text.txt"
            else {
                "abstract": paper.props.get("abstract") or "no abstract",
                "title": paper.props.get("title") or "no title",
            }
        )
        response = self._llm_client.prompt_file(
            user_prompt_file=os.path.join(os.path.dirname(__file__), "content", "prompts", paper_finding_txt),
            system_prompt="Extract field",
            params=parameters,
            prompt_settings={"prompt_tag": "paper_category", "temperature": 0.8},
        )

        if isinstance(response, str):
            result = response
        else:
            logger.warning(f"LLM failed to return a valid categorization response for {paper.id}: {response}")

        if result in self.CATEGORIES:
            return result

        return "other"

    async def _get_paper_categorizations(self, paper: Paper) -> str:
        """Categorize papers with multiple strategies and return the counts of each category."""
        # Print the PMID of the paper being categorized
        print(f"PMID: {paper.id}")

        # Categorize the paper by both keyword and LLM prompt.
        keyword_cat = self._get_keyword_category(paper)
        print("keyword_cat", keyword_cat)
        # TODO: uncomment the line below to use the chain of thought approach
        #llm_cat = await self._get_llm_category(paper)
        llm_cat = await self._apply_chain_of_thought(paper)
        print("llm_cat", llm_cat)
        # llm_cat = self._collect_neg_pos_papers(paper, all_papers)
        # llm_cat = self._few_shot_examples(gene, paper, all_papers)

        # If the keyword and LLM categories agree, just return that category.
        if keyword_cat == llm_cat:
            paper.props["disease_category"] = keyword_cat
            return keyword_cat

        counts: Dict[str, int] = {}
        # Otherwise it's conflicting - run the LLM prompt two more times and accumulate all the results.
        #llm_tiebreakers = await asyncio.gather(self._get_llm_category(paper), self._get_llm_category(paper))
        llm_tiebreakers = await asyncio.gather(self._apply_chain_of_thought(paper), self._apply_chain_of_thought(paper))
        for category in [keyword_cat, llm_cat, *llm_tiebreakers]:
            counts[category] = counts.get(category, 0) + 1
        assert len(counts) > 1 and sum(counts.values()) == 4

        best_category = max(counts, key=lambda k: counts[k])
        assert best_category in self.CATEGORIES and counts[best_category] < 4
        # Mark as conflicting if the best category has a low count.
        if counts[best_category] < 3:
            best_category = "conflicting"

        paper.props["disease_categorizations"] = counts
        paper.props["disease_category"] = best_category
        return best_category

    async def _get_all_papers(self, query: Dict[str, Any]) -> Sequence[Paper]:
        """Search for papers based on the given query.

        Args:
            query (Dict[str, Any]): The query to search for.

        Returns:
            Sequence[Paper]: The papers that match the query, tagged with a `disease_category` property representing
            what disease type it is judged to reference (rare disease, non-rare disease, other, and conflicting). If
            the paper is tagged as conflicting, it will also have a `disease_categorizations` property that shows the
            counts of the categorizations.
        """
        if not query.get("gene_symbol"):
            raise ValueError("Minimum requirement to search is to input a gene symbol.")
        params = {"query": query["gene_symbol"]}
        print("QUERY", query)
        # Rationalize the optional parameters.
        if ("max_date" in query or "date_type" in query) and "min_date" not in query:
            raise ValueError("A min_date is required when max_date or date_type is provided.")
        if "min_date" in query:
            params["min_date"] = query["min_date"]
            params["max_date"] = query.get("max_date", date.today().strftime("%Y/%m/%d"))
            params["date_type"] = query.get("date_type", "pdat")
        if "retmax" in query:
            params["retmax"] = query["retmax"]

        # Perform the search for papers
        paper_ids = self._paper_client.search(**params)

        # Extract the paper content that we care about (e.g. title, abstract, PMID, etc.)
        papers = [
            paper
            for paper_id in paper_ids
            if (paper := self._paper_client.fetch(paper_id, include_fulltext=True)) is not None
            and paper.props["fulltext_xml"] is not None
        ]
        logger.info(f"Categorizing {len(papers)} papers for {query['gene_symbol']}.")

        # # Categorize the papers.
        # for paper in papers:
        #     categories = await self._get_paper_categorizations(paper)  # Await the coroutine
        #     best_category = max(categories, key=lambda k: categories[k])
        #     assert best_category in self.CATEGORIES and categories[best_category] < 4

        #     # If there are multiple categories and the best one has a low count, mark it conflicting.
        #     if len(categories) > 1:
        #         # Always keep categorizations if there's more than one category.
        #         paper.props["disease_categorizations"] = categories
        #         # Mark as conflicting if the best category has a low count.
        #         if categories[best_category] < 3:
        #             best_category = "conflicting"

        #     paper.props["disease_category"] = best_category

        await asyncio.gather(*[self._get_paper_categorizations(paper) for paper in papers])
        return papers

    def get_papers(self, query: Dict[str, Any]) -> Sequence[Paper]:
        """Search for papers based on the given query.

        Args:
            query (Dict[str, Any]): The query to search for.

        Returns:
            Sequence[Paper]: The set of rare disease papers that match the query.
        """
        all_papers = asyncio.run(self._get_all_papers(query))
        return list(filter(lambda p: p.props["disease_category"] in self._allowed_categories, all_papers))
