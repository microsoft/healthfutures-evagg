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
        """Categorize papers based on LLM prompts."""
        response = await self._llm_client.prompt_file(
            user_prompt_file=os.path.join(os.path.dirname(__file__), "content", "prompts", "paper_finding.txt"),
            system_prompt="Extract field",
            params={
                "abstract": paper.props.get("abstract") or "no abstract",
                "title": paper.props.get("title") or "no title",
            },
            prompt_settings={"prompt_tag": "paper_category", "temperature": 0.8},
        )

        if isinstance(response, str):
            result = response
        else:
            logger.warning(f"LLM failed to return a valid categorization response for {paper.id}: {response}")

        if result in self.CATEGORIES:
            return result

        return "other"
    
    async def _get_llm_category_full_text(self, paper: Paper) -> str:
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
    
    async def _get_llm_category_few_shot(self, paper: Paper) -> str:
        """Categorize papers based on LLM prompts."""
        response = await self._llm_client.prompt_file(
            user_prompt_file=os.path.join(os.path.dirname(__file__), "content", "prompts", "paper_finding_few_shot.txt"),
            system_prompt="Extract field",
            params={
                "abstract": paper.props.get("abstract") or "no abstract",
                "title": paper.props.get("title") or "no title",
            },
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
            
            "Below are four few shot examples of papers that are classified as 'rare disease'. These are in no particular order:\n"

            "Gene: FBN2, PMID: 31316167\n"
            "Title: A clinical scoring system for congenital contractural arachnodactyly.\n"
            "Abstract: Congenital contractural arachnodactyly (CCA) is an autosomal dominant connective tissue disorder manifesting joint contractures, arachnodactyly, crumpled ears, and kyphoscoliosis as main features. Due to its rarity, rather aspecific clinical presentation, and overlap with other conditions including Marfan syndrome, the diagnosis is challenging, but important for prognosis and clinical management. CCA is caused by pathogenic variants in FBN2, encoding fibrillin-2, but locus heterogeneity has been suggested. We designed a clinical scoring system and diagnostic criteria to support the diagnostic process and guide molecular genetic testing.\n"

            "Gene: COG4, PMID: 19494034\n"
            "Title: Golgi function and dysfunction in the first COG4-deficient CDG type II patient.\n"
            "Abstract: The conserved oligomeric Golgi (COG) complex is a hetero-octameric complex essential for normal glycosylation and intra-Golgi transport. An increasing number of congenital disorder of glycosylation type II (CDG-II) mutations are found in COG subunits indicating its importance in glycosylation. We report a new CDG-II patient harbouring a p.R729W missense mutation in COG4 combined with a submicroscopical deletion. The resulting downregulation of COG4 expression additionally affects expression or stability of other lobe A subunits. Despite this, full complex formation was maintained albeit to a lower extent as shown by glycerol gradient centrifugation. Moreover, our data indicate that subunits are present in a cytosolic pool and full complex formation assists tethering preceding membrane fusion. By extending this study to four other known COG-deficient patients, we now present the first comparative analysis on defects in transport, glycosylation and Golgi ultrastructure in these patients. The observed structural and biochemical abnormalities correlate with the severity of the mutation, with the COG4 mutant being the mildest. All together our results indicate that intact COG complexes are required to maintain Golgi dynamics and its associated functions. According to the current CDG nomenclature, this newly identified deficiency is designated CDG-IIj.\n"

            "Gene: DNAJC7, PMID: 33193563\n"
            "Title: A Novel Potentially Pathogenic Rare Variant in the DNAJC7 Gene Identified in Amyotrophic Lateral Sclerosis Patients From Mainland China.\n"
            "Abstract: Variants in the DNAJC7 gene have been shown to be novel causes of amyotrophic lateral sclerosis (ALS). However, the contributions of DNAJC7 mutations in Asian ALS patients remain unclear. In this study, we screened rare pathogenic variants in the DNAJC7 gene in a cohort of 578 ALS patients from Mainland China. A novel, rare, putative pathogenic variant c.712A>G (p.R238G) was identified in one sporadic ALS patient. The carrier with this variant exhibited symptom onset at a relatively younger age and experienced rapid disease progression. Our results expand the pathogenic variant spectrum of DNAJC7 and indicate that variants in the DNAJC7 gene may also contribute to ALS in the Chinese population.\n"

            "Gene: MLH3, PMID: 15193445\n"
            "Title: No association between two MLH3 variants (S845G and P844L)and colorectal cancer risk.\n"
            "Abstract: Recently we identified a new variant, S845G, in the MLH3 gene in 7 out of 327 patients suspected of hereditary nonpolyposis colorectal cancer but not fulfilling the Amsterdam criteria and in 1 out of 188 control subjects. As this variant might play a role in causing sporadic colorectal cancer, we analyzed its prevalence in sporadic colorectal cancer patients. We analyzed a small part of exon 1 of the MLH3 gene, including the S845G variant, in germline DNA of 467 white sporadic colorectal cancer patients and 497 white controls. The S845G variant was detected in five patients and eight controls; the results thus indicate that this variant does not confer an increased colorectal cancer risk. Another variant (P844L) was clearly a polymorphism. Three other missense variants were rare and the sample size of the study was too small to conclude whether they are pathogenic. In conclusion, no association was observed between two MLH3 variants (P844L and S845G) and colorectal cancer risk.\n"
            
            "Below are four few shot examples of papers that are classified as 'other'. These are in no particular order:\n"
            
            "Gene: MLH3, PMID: 20308424\n"
            "Title: Mammalian BLM helicase is critical for integrating multiple pathways of meiotic recombination.\n"
            "Abstract: Bloom's syndrome (BS) is an autosomal recessive disorder characterized by growth retardation, cancer predisposition, and sterility. BS mutated (Blm), the gene mutated in BS patients, is one of five mammalian RecQ helicases. Although BLM has been shown to promote genome stability by assisting in the repair of DNA structures that arise during homologous recombination in somatic cells, less is known about its role in meiotic recombination primarily because of the embryonic lethality associated with Blm deletion. However, the localization of BLM protein on meiotic chromosomes together with evidence from yeast and other organisms implicates a role for BLM helicase in meiotic recombination events, prompting us to explore the meiotic phenotype of mice bearing a conditional mutant allele of Blm. In this study, we show that BLM deficiency does not affect entry into prophase I but causes severe defects in meiotic progression. This is exemplified by improper pairing and synapsis of homologous chromosomes and altered processing of recombination intermediates, resulting in increased chiasmata. Our data provide the first analysis of BLM function in mammalian meiosis and strongly argue that BLM is involved in proper pairing, synapsis, and segregation of homologous chromosomes; however, it is dispensable for the accumulation of recombination intermediates.\n"

            "Gene: ZNF423, PMID: 31234811\n"
            "Title: Master regulator analysis of paragangliomas carrying SDHx, VHL, or MAML3 genetic alterations.\n"
            "Abstract: Succinate dehydrogenase (SDH) loss and mastermind-like 3 (MAML3) translocation are two clinically important genetic alterations that correlate with increased rates of metastasis in subtypes of human paraganglioma and pheochromocytoma (PPGL) neuroendocrine tumors. Although hypotheses propose that succinate accumulation after SDH loss poisons dioxygenases and activates pseudohypoxia and epigenomic hypermethylation, it remains unclear whether these mechanisms account for oncogenic transcriptional patterns. Additionally, MAML3 translocation has recently been identified as a genetic alteration in PPGL, but is poorly understood. We hypothesize that a key to understanding tumorigenesis driven by these genetic alterations is identification of the transcription factors responsible for the observed oncogenic transcriptional changes.\n"

            "Gene: RNASEH1, PMID: 35711919\n"
            "Title: Case Report: Rare Homozygous RNASEH1 Mutations Associated With Adult-Onset Mitochondrial Encephalomyopathy and Multiple Mitochondrial DNA Deletions.\n"
            "Abstract: Mitochondrial DNA (mtDNA) maintenance disorders embrace a broad range of clinical syndromes distinguished by the evidence of mtDNA depletion and/or deletions in affected tissues. Among the nuclear genes associated with mtDNA maintenance disorders, RNASEH1 mutations produce a homogeneous phenotype, with progressive external ophthalmoplegia (PEO), ptosis, limb weakness, cerebellar ataxia, and dysphagia. The encoded enzyme, ribonuclease H1, is involved in mtDNA replication, whose impairment leads to an increase in replication intermediates resulting from mtDNA replication slowdown. Here, we describe two unrelated Italian probands (Patient 1 and Patient 2) affected by chronic PEO, ptosis, and muscle weakness. Cerebellar features and severe dysphagia requiring enteral feeding were observed in one patient. In both cases, muscle biopsy revealed diffuse mitochondrial abnormalities and multiple mtDNA deletions. A targeted next-generation sequencing analysis revealed the homozygous RNASEH1 mutations c.129-3C>G and c.424G>A in patients 1 and 2, respectively. The c.129-3C>G substitution has never been described as disease-related and resulted in the loss of exon 2 in Patient 1 muscle RNASEH1 transcript. Overall, we recommend implementing the use of high-throughput sequencing approaches in the clinical setting to reach genetic diagnosis in case of suspected presentations with impaired mtDNA homeostasis.\n"

            "Gene: TAPBP, PMID: 24159917\n"
            "Title: Gastrointestinal stromal tumors: a case-only analysis of single nucleotide polymorphisms and somatic mutations.\n"
            "Abstract: Gastrointestinal stromal tumors are rare soft tissue sarcomas that typically develop from mesenchymal cells with acquired gain-in-function mutations in KIT or PDGFRA oncogenes. These somatic mutations have been well-characterized, but little is known about inherited genetic risk factors. Given evidence that certain susceptibility loci and carcinogens are associated with characteristic mutations in other cancers, we hypothesized that these signature KIT or PDGFRA mutations may be similarly fundamental to understanding gastrointestinal stromal tumor etiology. Therefore, we examined associations between 522 single nucleotide polymorphisms and seven KIT or PDGFRA tumor mutations types. Candidate pathways included dioxin response, toxin metabolism, matrix metalloproteinase production, and immune and inflammatory response.\n"
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

    async def _get_paper_categorizations(self, paper: Paper) -> str:
        """Categorize papers with multiple strategies and return the counts of each category."""

        # Categorize the paper by both keyword and LLM prompt.
        keyword_cat = self._get_keyword_category(paper)
        
        # TODO: uncomment the line below to use the chain of thought or few shot approaches. Configure to be in yaml.
        # llm_cat = await self._get_llm_category(paper)
        # llm_cat = await self._apply_chain_of_thought(paper)
        llm_cat = await self._get_llm_category_few_shot(paper)
        
        # If the keyword and LLM categories agree, just return that category.
        if keyword_cat == llm_cat:
            paper.props["disease_category"] = keyword_cat
            return keyword_cat

        counts: Dict[str, int] = {}
        
        # TODO: uncomment the line below to use the chain of thought or few shot approaches. Configure to be in yaml.
        #llm_tiebreakers = await asyncio.gather(self._get_llm_category(paper), self._get_llm_category(paper))
        #llm_tiebreakers = await asyncio.gather(self._apply_chain_of_thought(paper), self._apply_chain_of_thought(paper))
        llm_tiebreakers = await asyncio.gather(self._get_llm_category_few_shot(paper), self._get_llm_category_few_shot(paper))
        
        # Otherwise it's conflicting - run the LLM prompt two more times and accumulate all the results.
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
