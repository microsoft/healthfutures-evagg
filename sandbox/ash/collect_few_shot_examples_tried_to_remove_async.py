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
import time
from datetime import datetime
from functools import cache
from typing import Any, Dict, List, Optional, Set

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
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types import CreateEmbeddingResponse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from lib.config import PydanticYamlModel
from lib.di import DiContainer
from lib.evagg.llm import aoai
from lib.evagg.ref import IPaperLookupClient

logger = logging.getLogger(__name__)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)


def get_git_commit_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


@cache
def get_lookup_client() -> IPaperLookupClient:
    ncbi_lookup: IPaperLookupClient = DiContainer().create_instance({"di_factory": "lib/config/ncbi_lookup.yaml"}, {})
    return ncbi_lookup


def get_paper_titles_abstracts(genes: Dict[str, Set[str]]) -> Dict[str, Dict[str, Dict[str, str]]]:
    client = get_lookup_client()
    gene_pmid_title_abstract = {}

    for gene, pmids in genes.items():
        gene_pmid_title_abstract[gene] = {}
        for pmid in pmids:
            try:
                paper = client.fetch(pmid)
                title = paper.props.get("title", "Unknown") if paper else "Unknown"
                abstract = paper.props.get("abstract", "Unknown") if paper else "Unknown"
                gene_pmid_title_abstract[gene][pmid] = {"title": title, "abstract": abstract}
            except Exception as e:
                print(f"Error getting title for paper {pmid}: {e}")
                gene_pmid_title_abstract[gene][pmid] = {"title": "Unknown", "abstract": "Unknown"}
    return gene_pmid_title_abstract


def add_abstract(gene_pmid_title: Dict[str, Dict[str, Dict[str, str]]]) -> Dict[str, Dict[str, Dict[str, str]]]:
    client = get_lookup_client()

    for gene, pmids in gene_pmid_title.items():
        for pmid in pmids:
            try:
                paper = client.fetch(pmid)
                abstract = paper.props.get("abstract", "Unknown") if paper else "Unknown"
                gene_pmid_title[gene][pmid]["abstract"] = abstract
            except Exception as e:
                print(f"Error getting abstract for paper {pmid}: {e}")
                gene_pmid_title[gene][pmid]["abstract"] = "Unknown"
    return gene_pmid_title


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


def k_set_automatically(X):
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, "euclidean"), axis=1)) / X.shape[0])

    # Calculate the distance of each distortion point from the line formed by the first and last points
    x1, y1 = 1, distortions[0]
    x2, y2 = 9, distortions[-1]
    distances = []
    for i in range(len(distortions)):
        x0 = i + 1
        y0 = distortions[i]
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
        distances.append(numerator / denominator)

    # The optimal number of clusters is the one that corresponds to the point of maximum distance
    optimal_k = distances.index(max(distances)) + 1

    return optimal_k


class OpenAIConfig(PydanticYamlModel):
    deployment: str
    endpoint: str
    api_key: str
    api_version: str
    max_parallel_requests: int = 0


def embeddings(inputs: List[str], embedding_settings: Optional[Dict[str, Any]] = None) -> Dict[str, List[float]]:
    settings = {"model": "text-embedding-ada-002", **(embedding_settings or {})}

    actual_embeddings = {}

    def run_single_embedding(input: str) -> int:
        result: CreateEmbeddingResponse = embeddings.create(input=[input], **settings)
        actual_embeddings[input] = result.data[0].embedding
        return result.usage.prompt_tokens

    start_overall = time.time()
    tokens = [run_single_embedding(input) for input in inputs]
    elapsed = time.time() - start_overall

    logger.info(f"{len(inputs)} embeddings produced in {elapsed:.2f} seconds using {sum(tokens)} tokens.")
    return actual_embeddings


def cluster_papers(gene_pmid_title_abstract_dict, k_means_clusters, pos_or_neg):
    papers = [
        (gene, pmid, data["title"], data["abstract"])
        for gene, pmids in gene_pmid_title_abstract_dict.items()
        for pmid, data in pmids.items()
    ]
    texts = [preprocess_text((title or "") + " " + (abstract or "")) for _, _, title, abstract in papers]

    # Leverage aoai embedding method to vectorize and embed papers
    vectorized = embeddings(texts)
    print("VECTORIZED", vectorized)

    vectorizer = (
        TfidfVectorizer()
    )  # TODO: ada02 embedding # https://github.com/jeremiahwander/ev-agg-exp/blob/780d6cba036d6dc8a46afd2acbdcb607db51dbbd/lib/evagg/llm/aoai.py#L134
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=k_means_clusters, random_state=args.seed)  # Adjust the number of clusters as needed
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
    output_file_path = os.path.join(args.outdir, f"paper_clusters_{pos_or_neg}.html")
    py.write_html(fig, output_file_path)

    return clusters


def save_clusters(outdir, clusters, pos_or_neg):
    """Save the clusters in a tsv of gene, pmid, cluster."""
    with open(os.path.join(outdir, f"paper_clusters_{pos_or_neg}.tsv"), "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["gene", "pmid", "cluster"])
        for cluster, papers in clusters.items():
            for gene, pmid in papers:
                writer.writerow([gene, pmid, cluster])


def sample_save_examples(seed, outdir, clusters, gene_pmid_title_abstract_dict, out_name):
    """Randomly choose 1 paper from each cluster and ensure that gene has not been sampled,
    then save the pmid, title, and abstract to a file."""
    random.seed(seed)
    with open(os.path.join(outdir, out_name), "w") as f:
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
                    f.write(f"Gene: {gene}, PMID: {pmid}\n")
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


# Point to an output run, and gather the gene and pmids from the irrelevant papers
def collect_neg_papers(benchmark_file) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Collect the negative example papers from the benchmarking results."""
    # Initialize an empty dictionary to hold the data
    irrelevant_gene_pmid_title = {}

    # Open the file and read its contents
    with open(benchmark_file, "r") as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("GENE:"):
                gene = line.split(":")[1].strip()
                irrelevant_gene_pmid_title[gene] = {}
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("GENE:"):
                    if lines[i].strip().startswith("Found E.A.") and "irrelevant" in lines[i]:
                        i += 1
                        while i < len(lines) and lines[i].strip().startswith("*"):
                            parts = lines[i].strip().split("*")
                            pmid = parts[2].strip()
                            title = parts[3].strip()
                            irrelevant_gene_pmid_title[gene][pmid] = {"title": title}
                            i += 1
                    else:
                        i += 1
            else:
                i += 1

    return irrelevant_gene_pmid_title
    # # Go through all genes in the data and gather their paper information, and create a separate dict of Dict(gene:Dict(pmid:title, abstract))
    # for gene, gene_data in data.items():
    #     print(f"Gene: {gene}")
    #     print(f"Gene data: {gene_data}")
    #     # for pmid, pmid_data in gene_data.items():
    #     #     if "title" in pmid_data:
    #     #         print(f"Gene: {gene}, PMID: {pmid}, Title: {pmid_data['title']}")
    # exit()
    # paper_ids = list(data["ACAT2"].keys())
    # papers = [
    #     paper
    #     for paper_id in paper_ids
    #     if (paper := self._paper_client.fetch(paper_id, include_fulltext=True)) is not None
    # ]
    # print("papers", papers)
    # if self._require_full_text:
    #     papers = [p for p in papers if p.props.get("full_text_xml")]

    # logger.warning(f"Categorizing {len(papers)} papers for {query['gene_symbol']}.")


def main(args):

    # Ensure that the output directory is created (and overwritten if it already exists)
    if os.path.isfile(args.outdir):
        os.remove(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    if args.fetch_paper_titles_abstracts:
        # Get ground truth data (gene and associated pmids into a dictionary) from
        gene_pmid_dict = parse_tsv(args.truth_file)

        # Get and then save the titles for these pmids.
        pos_gene_pmid_title_abstract = get_paper_titles_abstracts(gene_pmid_dict)

        with open(args.json_file_name, "w") as f:
            json.dump(pos_gene_pmid_title_abstract, f)

        with open(args.pickle_file_name, "wb") as f:
            pickle.dump(pos_gene_pmid_title_abstract, f)

    else:
        print("\nReading the truth data pickle file: ", args.pickle_file_name)

        with open(args.pickle_file_name, "rb") as f:
            pos_gene_pmid_title_abstract = pickle.load(f)

    # Ensure that you pair the .json and .pkl files with the few shot results
    shutil.copy(args.json_file_name, args.outdir)
    shutil.copy(args.pickle_file_name, args.outdir)

    print(f"\nProcessing truth data file to extract {args.k_means_clusters} positive examples...")

    # Determine the number of clusters
    # num_clusters = k_set_automatically()

    # Cluster (based on k) the positive example papers
    clusters = cluster_papers(pos_gene_pmid_title_abstract, args.k_means_clusters, "pos")

    # Save the clusters in a tsv of gene, pmid, cluster
    save_clusters(args.outdir, clusters, "pos")

    # Sample and save positive few shot examples
    sample_save_examples(args.seed, args.outdir, clusters, pos_gene_pmid_title_abstract, "few_shot_pos_examples.txt")

    # If a benchmark file is provided - we can process negative examples
    if args.benchmark_file:
        print(f"\nProcessing benchmark file to extract {args.k_means_clusters} negative examples...")

        # Collect negative (irrelevant) examples based on a pipeline run
        neg_dict = collect_neg_papers(args.benchmark_file)

        # Add the abstracts to those negative examples
        neg_dict = add_abstract(neg_dict)

        # Cluster (based on k) the negative example papers
        clusters = cluster_papers(neg_dict, args.k_means_clusters, "neg")

        # Save the clusters in a tsv of gene, pmid, cluster
        save_clusters(args.outdir, clusters, "neg")

        # Sample and save positive few shot examples
        sample_save_examples(args.seed, args.outdir, clusters, neg_dict, "few_shot_neg_examples.txt")


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
        default=4,
        type=int,
        help="Default is 4. This value must be an integer.",
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
        "-n",
        "--just-find-negatives",
        default=False,
        type=bool,
        help="Default is False. If True, only negative examples will be identified based on a benchmark file.",
    )
    parser.add_argument(
        "-b",
        "--benchmark-file",
        type=str,
        help=("Benchmark output file (benchmark_paper_finding_results... .txt). No default."),
    )  # e.g. /home/azureuser/ev-agg-exp/.out/binary_classes_paper_finding_results_2024-04-24/benchmarking_paper_finding_results_train.txt
    parser.add_argument(
        "-o",
        "--outdir",
        default=f".out/few_shot_examples_{(datetime.today().strftime('%Y-%m-%d'))}_{get_git_commit_hash()}/",
        type=str,
        help=(
            "Results output directory. Default is "
            f".out/few_shot_examples_{(datetime.today().strftime('%Y-%m-%d'))}_{get_git_commit_hash()}/"
        ),
    )
    args = parser.parse_args()

    if args.just_find_negatives and not args.benchmark_file:
        parser.error("-b/--benchmark-file is required when -n/--just-find-negatives is provided.")

    print("Evidence Aggregator Few Shot Example Generator is running with the following parameters:")
    for arg, value in vars(args).items():
        print(f"- {arg}: {value}")

    main(args)