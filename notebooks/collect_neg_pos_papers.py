import argparse
import csv

# Libraries
import glob
import json
import logging
import os
import pickle
import random
import shutil
import string
import subprocess
from datetime import datetime
from functools import cache
from typing import Dict, Set

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as py
import yaml
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from lib.di import DiContainer
from lib.evagg.ref import IPaperLookupClient

logger = logging.getLogger(__name__)
nltk.download("stopwords")
nltk.download("punkt")


def get_git_commit_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


@cache
def get_lookup_client() -> IPaperLookupClient:
    ncbi_lookup: IPaperLookupClient = DiContainer().create_instance({"di_factory": "lib/config/ncbi_lookup.yaml"}, {})
    return ncbi_lookup


def get_paper_titles(genes: Dict[str, Set[str]]) -> Dict[str, Dict[str, Dict[str, str]]]:
    client = get_lookup_client()
    result = {}

    for gene, pmids in genes.items():
        result[gene] = {}
        for pmid in pmids:
            try:
                paper = client.fetch(pmid)
                title = paper.props.get("title", "Unknown") if paper else "Unknown"
                abstract = paper.props.get("abstract", "Unknown") if paper else "Unknown"
                result[gene][pmid] = {"title": title, "abstract": abstract}
            except Exception as e:
                print(f"Error getting title for paper {pmid}: {e}")
                result[gene][pmid] = {"title": "Unknown", "abstract": "Unknown"}
    return result


def parse_tsv(file):
    with open(file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # Skip the header
        gene_dict = {}
        for row in reader:
            gene = row[0]
            pmid = row[1]
            if gene in gene_dict:
                gene_dict[gene].append(pmid)
            else:
                gene_dict[gene] = [pmid]
    return gene_dict


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords.words("english")]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(tokens)


def cluster_papers(gene_pmid_title_abstract_dict, k_means_clusters):
    papers = [
        (gene, pmid, data["title"], data["abstract"])
        for gene, pmids in gene_pmid_title_abstract_dict.items()
        for pmid, data in pmids.items()
    ]
    texts = [preprocess_text((title or "") + " " + (abstract or "")) for _, _, title, abstract in papers]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=k_means_clusters)  # Adjust the number of clusters as needed
    kmeans.fit(X)
    clusters = {i: [] for i in range(kmeans.n_clusters)}
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(papers[i][:2])  # (gene, pmid)

    # Visualize the clusters
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X.toarray())
    fig = go.Figure()
    for i in range(kmeans.n_clusters):
        points = X_3d[kmeans.labels_ == i]
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers+text",
                text=[gene for gene, _, _, _ in papers],
                marker=dict(size=6, line=dict(width=0.5), opacity=0.8),
            )
        )
    fig.update_layout(
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        width=700,
        margin=dict(r=20, b=10, l=10, t=10),
    )

    # Save the figure to an HTML file
    output_file_path = os.path.join(args.outdir, "paper_clusters.html")
    py.write_html(fig, output_file_path)

    return clusters


def main(args):

    # Ensure that output directory is created (and overwritten if it already exists)
    if os.path.isfile(args.outdir):
        os.remove(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    if args.fetch_paper_titles_abstracts:
        # Get ground truth data (gene and associated pmids into a dictionary) from
        gene_pmid_dict = parse_tsv(args.truth_file)

        # Cache the titles for these pmids.
        gene_pmid_title_abstract_dict = get_paper_titles(gene_pmid_dict)

        with open(args.json_file_name, "w") as f:
            json.dump(gene_pmid_title_abstract_dict, f)

        with open(args.pickle_file_name, "wb") as f:
            pickle.dump(gene_pmid_title_abstract_dict, f)
    else:
        logger.info("Reading the truth data pickle file: ", args.pickle_file_name)

        with open(args.pickle_file_name, "rb") as f:
            gene_pmid_title_abstract_dict = pickle.load(f)

    # Cluster the papers
    clusters = cluster_papers(gene_pmid_title_abstract_dict, args.k_means_clusters)

    # Save the clusters in a tsv of gene, pmid, cluster
    with open(os.path.join(args.outdir, "paper_clusters.tsv"), "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["gene", "pmid", "cluster"])
        for cluster, papers in clusters.items():
            for gene, pmid in papers:
                writer.writerow([gene, pmid, cluster])

    # Randomly choose 1 paper from each cluster and ensure that gene has not been sampled, then save the pmid, title, and abstract to a file
    random.seed(args.seed)
    with open(os.path.join(args.outdir, "few_shot_examples.txt"), "w") as f:
        sampled_genes = set()  # keep track of genes that have been sampled
        clustered_papers = list(clusters.items())  # convert dict to list for indexing
        i = 0  # start from the first cluster
        iterations = 0  # count the number of iterations
        while i < len(clustered_papers):
            if iterations > len(clustered_papers):  # if the number of iterations exceeds the number of clusters
                break  # break the loop
            cluster, papers = clustered_papers[i]
            random.shuffle(papers)  # shuffle the papers
            for j, (gene, pmid) in enumerate(papers):
                title = gene_pmid_title_abstract_dict[gene][pmid]["title"]
                abstract = gene_pmid_title_abstract_dict[gene][pmid]["abstract"]
                if (
                    gene not in sampled_genes and title is not None and abstract is not None
                ):  # if the gene has not been sampled and the paper has a title and abstract
                    f.write(f"Cluster: {cluster}, Gene: {gene}, PMID: {pmid}\n")
                    f.write(f"Title: {title}\n")
                    f.write(f"Abstract: {abstract}\n\n")
                    sampled_genes.add(gene)  # mark the gene as sampled
                    break
            else:  # if no paper with a unique gene was found in this cluster
                if i > 0:  # if this is not the first cluster
                    i -= 1  # go back to the previous cluster
                continue  # skip the increment of i
            i += 1  # move on to the next cluster
            iterations += 1  # increment the number of iterations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evidence Aggregator Few Shot Example Generator")
    parser.add_argument(
        "-t",
        "--truth-file",
        nargs="?",
        default="data/v1/papers_train_v1.tsv",
        type=str,
        help="Default is data/v1/papers_train_v1.tsv. This file format must be followed.",
    )
    parser.add_argument(
        "-j",
        "--json-file-name",
        nargs="?",
        default="data/v1/train_set_genes_pmids_titles_abstracts.json",
        type=str,
        help="Default is data/v1/train_set_genes_pmids_titles_abstracts.json. This file format must be followed.",
    )
    parser.add_argument(
        "-p",
        "--pickle-file-name",
        nargs="?",
        default="data/v1/train_set_genes_pmids_titles_abstracts.pkl",
        type=str,
        help="Default is data/v1/train_set_genes_pmids_titles_abstracts.pkl. This file format must be followed.",
    )
    parser.add_argument(
        "-f",
        "--fetch-paper-titles-abstracts",
        nargs="?",
        default=False,
        type=bool,
        help="Default is False. If True, fetches the paper titles and abstracts for the pmids in the truth file.",
    )
    parser.add_argument(
        "-k",
        "--k-means-clusters",
        nargs="?",
        default=5,
        type=int,
        help="Default is 5. This value must be an integer.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        nargs="?",
        default=0,
        type=int,
        help="Default is 0. This value must be an integer.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default=f"data/v1/few_shot_examples_{(datetime.today().strftime('%Y-%m-%d'))}_{get_git_commit_hash()}/",
        type=str,
        help=(
            "Results output directory. Default is "
            f"data/v1/few_shot_examples_{(datetime.today().strftime('%Y-%m-%d'))}_{get_git_commit_hash()}/"
        ),
    )
    args = parser.parse_args()

    main(args)

############################################################################################################
# def get_llm_category_w_few_shot(self, paper: Paper) -> str:
#     paper_finding_txt = (
#         "paper_finding.txt" if paper.props.get("full_text_xml") is None else "paper_finding_full_text.txt"
#     )
#     parameters = (
#         self._get_paper_texts(paper)
#         if paper_finding_txt == "paper_finding_full_text.txt"
#         else {
#             "abstract": paper.props.get("abstract") or "no abstract",
#             "title": paper.props.get("title") or "no title",
#         }
#     )
#     response = self._llm_client.prompt_file(
#         user_prompt_file=os.path.join(os.path.dirname(__file__), "content", "prompts", paper_finding_txt),
#         system_prompt="Extract field",
#         params=parameters,
#         prompt_settings={"prompt_tag": "paper_category", "temperature": 0.8},
#     )

#     if isinstance(response, str):
#         result = response
#     else:
#         logger.warning(f"LLM failed to return a valid categorization response for {paper.id}: {response}")

#     if result in self.CATEGORIES:
#         return result

#     return "other"


# # papers_dict = {paper.id: paper for paper in papers}
# # print("papers_dict", papers_dict['pmid:31839819'])

# # # Check if papers_dict['pmid:31839819'] is the same as paper1
# # paper1_title = paper1.props.get("title")
# # paper1_abstract = paper1.props.get("abstract")
# # print("paper1_title", paper1_title)
# # print("paper1_abstract", paper1_abstract)
# # print("paper1", paper1)
# # print("papers_dict['pmid:31839819']", papers_dict['pmid:31839819'])
# # print("papers_dict['pmid:31839819'].props.get('title')", papers_dict['pmid:31839819'].props.get("title"))
# # print("papers_dict['pmid:31839819'].props.get('abstract')", papers_dict['pmid:31839819'].props.get("abstract"))
# # print("papers_dict['pmid:31839819'].props.get('title') == paper1_title", papers_dict['pmid:31839819'].props.get("title") == paper1_title)
# # print("papers_dict['pmid:31839819'].props.get('abstract') == paper1_abstract", papers_dict['pmid:31839819'].props.get("abstract") == paper1_abstract)
# # print("papers_dict['pmid:31839819'] == paper1", papers_dict['pmid:31839819'] == paper1)


# def collect_neg_pos_papers(self, paper1, papers) -> None:
#     """Collect the negative example papers from the benchmarking results, and positive example papers from the truth set."""
#     # Define the path to the file
#     irrelevant_paper_file_path = "/home/azureuser/ev-agg-exp/.out/binary_classes_paper_finding_results_2024-04-24/benchmarking_paper_finding_results_train.txt"
#     truth_set_file_path = "/home/azureuser/ev-agg-exp/data/v1/papers_train_v1.tsv"

#     # Initialize an empty dictionary to hold the data
#     data = {}

#     # Open the file and read its contents
#     with open(irrelevant_paper_file_path, "r") as file:
#         lines = file.readlines()
#         i = 0
#         while i < len(lines):
#             line = lines[i].strip()
#             if line.startswith("GENE:"):
#                 gene = line.split(":")[1].strip()
#                 data[gene] = {}
#                 i += 1
#                 while i < len(lines) and not lines[i].strip().startswith("GENE:"):
#                     if lines[i].strip().startswith("Found E.A.") and "irrelevant" in lines[i]:
#                         i += 1
#                         while i < len(lines) and lines[i].strip().startswith("*"):
#                             parts = lines[i].strip().split("*")
#                             pmid = parts[2].strip()
#                             title = parts[3].strip()
#                             data[gene][pmid] = {"title": title}
#                             i += 1
#                     else:
#                         i += 1
#             else:
#                 i += 1

#     # Print the data
#     # Go through all genes in the data and gather their paper information, and create a separate dict of Dict(gene:Dict(pmid:title, abstract))
#     for gene, gene_data in data.items():
#         print(f"Gene: {gene}")
#         print(f"Gene data: {gene_data}")
#         # for pmid, pmid_data in gene_data.items():
#         #     if "title" in pmid_data:
#         #         print(f"Gene: {gene}, PMID: {pmid}, Title: {pmid_data['title']}")
#     exit()
#     paper_ids = list(data["ACAT2"].keys())
#     papers = [
#         paper
#         for paper_id in paper_ids
#         if (paper := self._paper_client.fetch(paper_id, include_fulltext=True)) is not None
#     ]
#     print("papers", papers)
#     if self._require_full_text:
#         papers = [p for p in papers if p.props.get("full_text_xml")]

#     logger.warning(f"Categorizing {len(papers)} papers for {query['gene_symbol']}.")
#     exit()


# def get_example_type_and_gene(self, pmid) -> Tuple[str, str]:
#     # Read the data from the TSV file into a pandas DataFrame
#     df = pd.read_csv("/home/azureuser/ev-agg-exp/data/v1/papers_train_v1.tsv", sep="\t")

#     # Get the row for the given PMID
#     row = df[df["pmid"] == pmid]
#     print("row", row)

#     # If the row is empty, the example is negative and the gene is None
#     if row.empty:
#         return "negative", "NA"

#     # Otherwise, the example is positive and the gene is the value in the "gene" column
#     return "positive", row["gene"].values[0]


# def few_shot_examples(self, gene, paper: Paper, papers: Sequence[Paper]) -> Sequence[Paper]:
#     """Given the paper (title and abstract in question), compute the cosine similarity between that paper and the
#     other papers in the dataset. Return the top 2 most similar papers in the positive category, and 2 most similar papers in the negative category.
#     """
#     # Extract the title and abstract of the paper in question
#     title = paper.props.get("title") or ""
#     abstract = paper.props.get("abstract") or ""

#     # Extract the titles and abstracts of all the papers in the dataset TODO: do not need to get all of them
#     dataset_example_signs_and_genes = [self._get_example_type_and_gene(p.id) for p in papers]
#     print(dataset_example_signs_and_genes)

#     # Separate the positive and negative papers, excluding papers with the same gene as the paper in question
#     positive_papers = [
#         p for p, (sign, g) in zip(papers, dataset_example_signs_and_genes) if sign == "positive" and g != gene
#     ]
#     negative_papers = [
#         p for p, (sign, g) in zip(papers, dataset_example_signs_and_genes) if sign == "negative" and g != gene
#     ]

#     # Combine the titles and abstracts into separate lists of documents for positive and negative papers
#     positive_documents = (
#         [title]
#         + [abstract]
#         + [p.props.get("title") or "" for p in positive_papers]
#         + [p.props.get("abstract") or "" for p in positive_papers]
#     )
#     negative_documents = (
#         [title]
#         + [abstract]
#         + [p.props.get("title") or "" for p in negative_papers]
#         + [p.props.get("abstract") or "" for p in negative_papers]
#     )

#     # Create a TF-IDF vectorizer and transform the documents into TF-IDF vectors
#     vectorizer = TfidfVectorizer()
#     positive_tfidf_matrix = vectorizer.fit_transform(positive_documents)
#     negative_tfidf_matrix = vectorizer.fit_transform(negative_documents)

#     # Compute the cosine similarity matrix between the TF-IDF vectors
#     positive_similarity_matrix = cosine_similarity(positive_tfidf_matrix)
#     negative_similarity_matrix = cosine_similarity(negative_tfidf_matrix)

#     # Get the indices of the top 2 most similar papers (excluding the paper in question) for both positive and negative papers
#     paper_index = 0
#     positive_top_indices = positive_similarity_matrix[paper_index].argsort()[-3:-1][::-1]
#     negative_top_indices = negative_similarity_matrix[paper_index].argsort()[-3:-1][::-1]

#     # Return the top 2 most similar papers for both positive and negative papers
#     positive_similar_papers = [positive_papers[i] for i in positive_top_indices]
#     negative_similar_papers = [negative_papers[i] for i in negative_top_indices]

#     # Save the top 4 similar (neg and pos categories) to a file, labeled with the anchor paper's PMID and gene name
#     with open(f"lib/evagg/content/prompts/similar_papers_{paper.id.replace('pmid:', '')}_{gene}.txt", "w") as f:
#         f.write(
#             "Two example papers from the 'rare disease' category that are similar to the current paper we are trying to classify:\n"
#         )
#         for p in positive_similar_papers:
#             f.write(f"Title: {p.props.get('title')}\n")
#             f.write(f"Abstract: {p.props.get('abstract')}\n\n")
#         f.write(
#             "Two example papers from the 'other' category that are similar to the current paper we are trying to classify:\n"
#         )
#         for p in negative_similar_papers:
#             f.write(f"Title: {p.props.get('title')}\n")
#             f.write(f"Abstract: {p.props.get('abstract')}\n\n")

#     return positive_similar_papers, negative_similar_papers


# # Define the path to the file
# file_path = "/home/azureuser/ev-agg-exp/.out/binary_classes_paper_finding_results_2024-04-24/benchmarking_paper_finding_results_train.txt"

# # Initialize an empty dictionary to hold the data
# data = {}

# # Open the file and read its contents
# with open(file_path, "r") as file:
#     lines = file.readlines()
#     i = 0
#     while i < len(lines):
#         line = lines[i].strip()
#         if line.startswith("GENE:"):
#             gene = line.split(":")[1].strip()
#             data[gene] = {}
#             i += 1
#             while i < len(lines) and not lines[i].strip().startswith("GENE:"):
#                 if lines[i].strip().startswith("Found E.A.") and "irrelevant" in lines[i]:
#                     i += 1
#                     while i < len(lines) and lines[i].strip().startswith("*"):
#                         parts = lines[i].strip().split("*")
#                         pmid = parts[2].strip()
#                         title = parts[3].strip()
#                         data[gene][pmid] = {"title": title}
#                         i += 1
#                 else:
#                     i += 1
#         else:
#             i += 1

# # Print the data
# print(data["ACAT2"])

# # main
# queries = [
#     {"gene_symbol": "ADCY1", "min_date": "2014/01/01", "retmax": 3},  # 99
#     # {"gene_symbol": "RHOH", "min_date": "2012/01/01", "retmax": 90},
#     # {"gene_symbol": "FBN2", "min_date": "1998/01/01", "retmax": 313},
#     # {"gene_symbol": "ACAT2", "min_date": "2021/01/01", "retmax": 87},
#     # {"gene_symbol": "TAPBP", "min_date": "2002/01/01", "retmax": 65},
#     # {"gene_symbol": "LRRC10", "min_date": "2015/01/01", "retmax": 24},
#     # {"gene_symbol": "EXOC2", "min_date": "2020/01/01", "retmax": 22},
#     # {"gene_symbol": "CTF1", "min_date": "2000/01/01", "retmax": 115},
#     # {"gene_symbol": "TOPBP1", "min_date": "2014/01/01", "retmax": 260},
#     # {"gene_symbol": "TNNC2", "min_date": "2023/01/01", "retmax": 19},
#     # {"gene_symbol": "PEX11A", "min_date": "2020/01/01", "retmax": 18},
#     # {"gene_symbol": "KMO", "min_date": "2023/01/01", "retmax": 170},
#     # {"gene_symbol": "GRXCR2", "min_date": "2014/01/01", "retmax": 19},
#     # {"gene_symbol": "CPT1B", "min_date": "2021/01/01", "retmax": 152},
#     # {"gene_symbol": "COG4", "min_date": "2009/01/01", "retmax": 63},
#     # {"gene_symbol": "MLH3", "min_date": "2001/01/01", "retmax": 273},
#     # {"gene_symbol": "HYAL1", "min_date": "1999/01/01", "retmax": 316},
#     # {"gene_symbol": "EMC1", "min_date": "2016/01/01", "retmax": 37},
#     # {"gene_symbol": "TOP2B", "min_date": "2019/01/01", "retmax": 102},
#     # {"gene_symbol": "OTUD7A", "min_date": "2020/01/01", "retmax": 26},
#     # {"gene_symbol": "DNAJC7", "min_date": "2019/01/01", "retmax": 34},
#     # {"gene_symbol": "SARS1", "min_date": "2017/01/01", "retmax": 46},
#     # {"gene_symbol": "NPPA", "min_date": "2020/01/01", "retmax": 233},
#     # {"gene_symbol": "RNASEH1", "min_date": "2019/01/01", "retmax": 62},
#     # {"gene_symbol": "IGKC", "min_date": "1976/01/01", "retmax": 122},
#     # {"gene_symbol": "RGS9", "min_date": "2004/01/01", "retmax": 185},
#     # {"gene_symbol": "SLFN14", "min_date": "2015/01/01", "retmax": 28},
#     # {"gene_symbol": "SLC38A9", "min_date": "2022/01/01", "retmax": 29},
#     # {"gene_symbol": "B4GAT1", "min_date": "2013/01/01", "retmax": 29},
#     # {"gene_symbol": "ZNF423", "min_date": "2012/01/01", "retmax": 80},
#     # {"gene_symbol": "BAZ2B", "min_date": "2020/01/01", "retmax": 32},
#     # {"gene_symbol": "JPH2", "min_date": "2007/01/01", "retmax": 117},
# ]


# def _call_approach_comparisons(self, paper: Paper, query) -> str:
#     # TODO: comparing approaches
#     # Call the function to compute the cosine similarity
#     all_papers = self._get_all_papers(query)
#     print("all_papers", all_papers)
#     # similar_papers = self._few_shot_examples(paper, all_papers)
#     # print("similar_papers", similar_papers)
#     exit()
