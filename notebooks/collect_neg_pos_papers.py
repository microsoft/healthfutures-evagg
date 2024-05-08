import logging
import subprocess
from typing import Any, Dict, Sequence

from lib.evagg.types import Paper

logger = logging.getLogger(__name__)


def get_git_commit_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()






def get_all_papers(query: Dict[str, Any]) -> Sequence[Paper]:
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
    paper_ids = paper_client.search(**params)

    # Extract the paper content that we care about (e.g. title, abstract, PMID, etc.)
    papers = [
        paper
        for paper_id in paper_ids
        if (paper := self._paper_client.fetch(paper_id, include_fulltext=True)) is not None
    ]
    print("papers", papers)
    if self._require_full_text:
        papers = [p for p in papers if p.props.get("full_text_xml")]

    logger.warning(f"Categorizing {len(papers)} papers for {query['gene_symbol']}.")

    # Categorize the papers.
    for paper in papers:
        categories = self._get_paper_categorizations(query["gene_symbol"], paper, papers)
        best_category = max(categories, key=lambda k: categories[k])
        assert best_category in self.CATEGORIES and categories[best_category] < 4

        # If there are multiple categories and the best one has a low count, mark it conflicting.
        if len(categories) > 1:
            # Always keep categorizations if there's more than one category.
            paper.props["disease_categorizations"] = categories
            # Mark as conflicting if the best category has a low count.
            if categories[best_category] < 3:
                best_category = "conflicting"

        paper.props["disease_category"] = best_category

    return papers


# Define the path to the file
file_path = "/home/azureuser/ev-agg-exp/.out/binary_classes_paper_finding_results_2024-04-24/benchmarking_paper_finding_results_train.txt"

# Initialize an empty dictionary to hold the data
data = {}

# Open the file and read its contents
with open(file_path, "r") as file:
    lines = file.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("GENE:"):
            gene = line.split(":")[1].strip()
            data[gene] = {}
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("GENE:"):
                if lines[i].strip().startswith("Found E.A.") and "irrelevant" in lines[i]:
                    i += 1
                    while i < len(lines) and lines[i].strip().startswith("*"):
                        parts = lines[i].strip().split("*")
                        pmid = parts[2].strip()
                        title = parts[3].strip()
                        data[gene][pmid] = {"title": title}
                        i += 1
                else:
                    i += 1
        else:
            i += 1

# Print the data
print(data["ACAT2"])

# main
queries = [
    {"gene_symbol": "ADCY1", "min_date": "2014/01/01", "retmax": 3},  # 99
    # {"gene_symbol": "RHOH", "min_date": "2012/01/01", "retmax": 90},
    # {"gene_symbol": "FBN2", "min_date": "1998/01/01", "retmax": 313},
    # {"gene_symbol": "ACAT2", "min_date": "2021/01/01", "retmax": 87},
    # {"gene_symbol": "TAPBP", "min_date": "2002/01/01", "retmax": 65},
    # {"gene_symbol": "LRRC10", "min_date": "2015/01/01", "retmax": 24},
    # {"gene_symbol": "EXOC2", "min_date": "2020/01/01", "retmax": 22},
    # {"gene_symbol": "CTF1", "min_date": "2000/01/01", "retmax": 115},
    # {"gene_symbol": "TOPBP1", "min_date": "2014/01/01", "retmax": 260},
    # {"gene_symbol": "TNNC2", "min_date": "2023/01/01", "retmax": 19},
    # {"gene_symbol": "PEX11A", "min_date": "2020/01/01", "retmax": 18},
    # {"gene_symbol": "KMO", "min_date": "2023/01/01", "retmax": 170},
    # {"gene_symbol": "GRXCR2", "min_date": "2014/01/01", "retmax": 19},
    # {"gene_symbol": "CPT1B", "min_date": "2021/01/01", "retmax": 152},
    # {"gene_symbol": "COG4", "min_date": "2009/01/01", "retmax": 63},
    # {"gene_symbol": "MLH3", "min_date": "2001/01/01", "retmax": 273},
    # {"gene_symbol": "HYAL1", "min_date": "1999/01/01", "retmax": 316},
    # {"gene_symbol": "EMC1", "min_date": "2016/01/01", "retmax": 37},
    # {"gene_symbol": "TOP2B", "min_date": "2019/01/01", "retmax": 102},
    # {"gene_symbol": "OTUD7A", "min_date": "2020/01/01", "retmax": 26},
    # {"gene_symbol": "DNAJC7", "min_date": "2019/01/01", "retmax": 34},
    # {"gene_symbol": "SARS1", "min_date": "2017/01/01", "retmax": 46},
    # {"gene_symbol": "NPPA", "min_date": "2020/01/01", "retmax": 233},
    # {"gene_symbol": "RNASEH1", "min_date": "2019/01/01", "retmax": 62},
    # {"gene_symbol": "IGKC", "min_date": "1976/01/01", "retmax": 122},
    # {"gene_symbol": "RGS9", "min_date": "2004/01/01", "retmax": 185},
    # {"gene_symbol": "SLFN14", "min_date": "2015/01/01", "retmax": 28},
    # {"gene_symbol": "SLC38A9", "min_date": "2022/01/01", "retmax": 29},
    # {"gene_symbol": "B4GAT1", "min_date": "2013/01/01", "retmax": 29},
    # {"gene_symbol": "ZNF423", "min_date": "2012/01/01", "retmax": 80},
    # {"gene_symbol": "BAZ2B", "min_date": "2020/01/01", "retmax": 32},
    # {"gene_symbol": "JPH2", "min_date": "2007/01/01", "retmax": 117},
]
for query in queries:
    res = get_all_papers(query)
















def _get_llm_category_w_few_shot(self, paper: Paper) -> str:
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

 # papers_dict = {paper.id: paper for paper in papers}
        # print("papers_dict", papers_dict['pmid:31839819'])
        
        # # Check if papers_dict['pmid:31839819'] is the same as paper1
        # paper1_title = paper1.props.get("title")
        # paper1_abstract = paper1.props.get("abstract")
        # print("paper1_title", paper1_title)
        # print("paper1_abstract", paper1_abstract)
        # print("paper1", paper1)
        # print("papers_dict['pmid:31839819']", papers_dict['pmid:31839819'])
        # print("papers_dict['pmid:31839819'].props.get('title')", papers_dict['pmid:31839819'].props.get("title"))
        # print("papers_dict['pmid:31839819'].props.get('abstract')", papers_dict['pmid:31839819'].props.get("abstract"))
        # print("papers_dict['pmid:31839819'].props.get('title') == paper1_title", papers_dict['pmid:31839819'].props.get("title") == paper1_title)
        # print("papers_dict['pmid:31839819'].props.get('abstract') == paper1_abstract", papers_dict['pmid:31839819'].props.get("abstract") == paper1_abstract)
        # print("papers_dict['pmid:31839819'] == paper1", papers_dict['pmid:31839819'] == paper1)
        
    def _collect_neg_pos_papers(self, paper1, papers) -> None:
        """Collect the negative example papers from the benchmarking results, and positive example papers from the truth set."""
        # Define the path to the file
        irrelevant_paper_file_path = "/home/azureuser/ev-agg-exp/.out/binary_classes_paper_finding_results_2024-04-24/benchmarking_paper_finding_results_train.txt"
        truth_set_file_path = "/home/azureuser/ev-agg-exp/data/v1/papers_train_v1.tsv"
            
        # Initialize an empty dictionary to hold the data
        data = {}

        # Open the file and read its contents
        with open(irrelevant_paper_file_path, "r") as file:
            lines = file.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("GENE:"):
                    gene = line.split(":")[1].strip()
                    data[gene] = {}
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith("GENE:"):
                        if lines[i].strip().startswith("Found E.A.") and "irrelevant" in lines[i]:
                            i += 1
                            while i < len(lines) and lines[i].strip().startswith("*"):
                                parts = lines[i].strip().split("*")
                                pmid = parts[2].strip()
                                title = parts[3].strip()
                                data[gene][pmid] = {"title": title}
                                i += 1
                        else:
                            i += 1
                else:
                    i += 1

        # Print the data
        # Go through all genes in the data and gather their paper information, and create a separate dict of Dict(gene:Dict(pmid:title, abstract))
        for gene, gene_data in data.items():
            print(f"Gene: {gene}")
            print(f"Gene data: {gene_data}")
            # for pmid, pmid_data in gene_data.items():
            #     if "title" in pmid_data:
            #         print(f"Gene: {gene}, PMID: {pmid}, Title: {pmid_data['title']}")
        exit()
        paper_ids = list(data["ACAT2"].keys())
        papers = [
            paper
            for paper_id in paper_ids
            if (paper := self._paper_client.fetch(paper_id, include_fulltext=True)) is not None
        ]
        print("papers", papers)
        if self._require_full_text:
            papers = [p for p in papers if p.props.get("full_text_xml")]

        logger.warning(f"Categorizing {len(papers)} papers for {query['gene_symbol']}.")
        exit()
    
    def _get_example_type_and_gene(self, pmid) -> Tuple[str, str]:
        # Read the data from the TSV file into a pandas DataFrame
        df = pd.read_csv("/home/azureuser/ev-agg-exp/data/v1/papers_train_v1.tsv", sep="\t")

        # Get the row for the given PMID
        row = df[df["pmid"] == pmid]
        print("row", row)

        # If the row is empty, the example is negative and the gene is None
        if row.empty:
            return "negative", "NA"

        # Otherwise, the example is positive and the gene is the value in the "gene" column
        return "positive", row["gene"].values[0]
            
    def _few_shot_examples(self, gene, paper: Paper, papers: Sequence[Paper]) -> Sequence[Paper]:
        """Given the paper (title and abstract in question), compute the cosine similarity between that paper and the
        other papers in the dataset. Return the top 2 most similar papers in the positive category, and 2 most similar papers in the negative category."""
        # Extract the title and abstract of the paper in question
        title = paper.props.get("title") or ""
        abstract = paper.props.get("abstract") or ""
        
        # Extract the titles and abstracts of all the papers in the dataset TODO: do not need to get all of them
        dataset_example_signs_and_genes = [self._get_example_type_and_gene(p.id) for p in papers]
        print(dataset_example_signs_and_genes)

        # Separate the positive and negative papers, excluding papers with the same gene as the paper in question
        positive_papers = [p for p, (sign, g) in zip(papers, dataset_example_signs_and_genes) if sign == "positive" and g != gene]
        negative_papers = [p for p, (sign, g) in zip(papers, dataset_example_signs_and_genes) if sign == "negative" and g != gene]

        # Combine the titles and abstracts into separate lists of documents for positive and negative papers
        positive_documents = [title] + [abstract] + [p.props.get("title") or "" for p in positive_papers] + [p.props.get("abstract") or "" for p in positive_papers]
        negative_documents = [title] + [abstract] + [p.props.get("title") or "" for p in negative_papers] + [p.props.get("abstract") or "" for p in negative_papers]

        # Create a TF-IDF vectorizer and transform the documents into TF-IDF vectors
        vectorizer = TfidfVectorizer()
        positive_tfidf_matrix = vectorizer.fit_transform(positive_documents)
        negative_tfidf_matrix = vectorizer.fit_transform(negative_documents)

        # Compute the cosine similarity matrix between the TF-IDF vectors
        positive_similarity_matrix = cosine_similarity(positive_tfidf_matrix)
        negative_similarity_matrix = cosine_similarity(negative_tfidf_matrix)

        # Get the indices of the top 2 most similar papers (excluding the paper in question) for both positive and negative papers
        paper_index = 0
        positive_top_indices = positive_similarity_matrix[paper_index].argsort()[-3:-1][::-1]
        negative_top_indices = negative_similarity_matrix[paper_index].argsort()[-3:-1][::-1]

        # Return the top 2 most similar papers for both positive and negative papers
        positive_similar_papers = [positive_papers[i] for i in positive_top_indices]
        negative_similar_papers = [negative_papers[i] for i in negative_top_indices]
        
        # Save the top 4 similar (neg and pos categories) to a file, labeled with the anchor paper's PMID and gene name
        with open(
            f"lib/evagg/content/prompts/similar_papers_{paper.id.replace('pmid:', '')}_{gene}.txt",
            "w"
        ) as f:
            f.write("Two example papers from the 'rare disease' category that are similar to the current paper we are trying to classify:\n")
            for p in positive_similar_papers:
                f.write(f"Title: {p.props.get('title')}\n")
                f.write(f"Abstract: {p.props.get('abstract')}\n\n")
            f.write("Two example papers from the 'other' category that are similar to the current paper we are trying to classify:\n")
            for p in negative_similar_papers:
                f.write(f"Title: {p.props.get('title')}\n")
                f.write(f"Abstract: {p.props.get('abstract')}\n\n")
        
        return positive_similar_papers, negative_similar_papers

    # def _call_approach_comparisons(self, paper: Paper, query) -> str:
    #     # TODO: comparing approaches
    #     # Call the function to compute the cosine similarity
    #     all_papers = self._get_all_papers(query)
    #     print("all_papers", all_papers)
    #     # similar_papers = self._few_shot_examples(paper, all_papers)
    #     # print("similar_papers", similar_papers)
    #     exit()