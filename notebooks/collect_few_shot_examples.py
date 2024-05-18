import argparse
import asyncio
import csv
import json
import logging
import os
import pickle
import random
import shutil
import subprocess
from datetime import datetime
from functools import cache
from typing import Dict, Set

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as py
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from lib.di import DiContainer
from lib.evagg.llm import OpenAIClient
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.svc import get_dotenv_settings

logger = logging.getLogger(__name__)


def get_git_commit_hash():
    """Get the git commit hash."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


@cache
def get_lookup_client() -> IPaperLookupClient:
    """Get the lookup client."""
    ncbi_lookup: IPaperLookupClient = DiContainer().create_instance({"di_factory": "lib/config/ncbi_lookup.yaml"}, {})
    return ncbi_lookup


def get_paper_titles_abstracts(genes: Dict[str, Set[str]]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Get the paper titles and abstracts for the given genes and pmids. This is used to get the positive examples."""
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
    """Add the abstract to the gene_pmid_title dictionary. This is used to get the negative examples."""
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
    """Parse input TSV file and return a dictionary of gene and associated pmids."""
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


async def get_papers_and_embeddings(gene_pmid_title_abstract_dict):
    """Get the papers and their embeddings."""
    papers = [
        (gene, pmid, data["title"], data["abstract"])
        for gene, pmids in gene_pmid_title_abstract_dict.items()
        for pmid, data in pmids.items()
    ]
    texts = [(title or "") + " " + (abstract or "") for _, _, title, abstract in papers]

    settings = get_dotenv_settings(filter_prefix="AZURE_OPENAI_")
    client = OpenAIClient(settings)
    embeddings = await client.embeddings(texts)
    embeddings = dict(sorted(embeddings.items()))

    texts = list(embeddings.keys())
    embedding_values = list(embeddings.values())
    return papers, texts, embedding_values


def plot_elbow(ssd, optimal_k_elbow, pos_or_neg, outdir):
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, 15), ssd, "bx-")
    plt.xlabel("k")
    plt.ylabel("Sum_of_squared_distances")
    plt.title("Elbow Method For Optimal k")
    plt.axvline(x=optimal_k_elbow, color="r", linestyle="--")
    plt.savefig(os.path.join(outdir, f"elbow_method_{pos_or_neg}.png"))
    plt.clf()


def determine_optimal_k_elbow(embedding_values):
    """Determine the optimal number of clusters using the elbow method."""
    ssd = []
    large_k = range(1, 15)
    for k in large_k:
        kmeans_k = KMeans(n_clusters=k, random_state=args.seed)
        kmeans_k = kmeans_k.fit(embedding_values)
        ssd.append(kmeans_k.inertia_)

    second_derivative = [0] + [ssd[i + 1] + ssd[i - 1] - 2 * ssd[i] for i in range(1, len(ssd) - 1)] + [0]
    optimal_k = second_derivative.index(max(second_derivative))
    print(f"Optimal k according to elbow method: {optimal_k}. Please also see 'elbow_method.png' for plot.")
    return optimal_k, ssd


def determine_optimal_k_silhouette(embedding_values_array):
    """Determine the optimal number of clusters using the silhouette method."""
    sil_scores = []
    large_k = range(2, 15)  # this method needs at least 2 clusters
    for k in large_k:
        kmeans_k = KMeans(n_clusters=k, random_state=args.seed)
        kmeans_k = kmeans_k.fit(embedding_values_array)
        score = silhouette_score(embedding_values_array, kmeans_k.labels_)
        sil_scores.append(score)

    optimal_k = large_k[sil_scores.index(max(sil_scores))]
    print(f"Optimal k according to silhouette method: {optimal_k}")


def determine_optimal_k_bic(embedding_values_array):
    """Determine the optimal number of clusters using the Bayesian Information Criterion (BIC) method."""
    bic = []
    large_k = range(1, 15)
    for k in large_k:
        gmm = GaussianMixture(n_components=k, random_state=args.seed)
        gmm.fit(embedding_values_array)
        bic.append(gmm.bic(embedding_values_array))

    optimal_k = large_k[bic.index(min(bic))]
    print(f"Optimal k according to BIC method: {optimal_k}")


def save_clusters_to_dict(embedding_values, papers, k_means_clusters):
    """Save the clusters and their kmeans distances."""
    kmeans = KMeans(n_clusters=k_means_clusters, random_state=args.seed)
    kmeans.fit(np.array(embedding_values))
    clusters = {i: [] for i in range(k_means_clusters)}
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(papers[i][:2])

    return clusters, kmeans


def visualize_clusters(embedding_values, papers, clusters, k_means_clusters, kmeans, pos_or_neg):
    """Visualize the clusters in 3D space."""
    pca = PCA(n_components=3)
    x_3d = pca.fit_transform(np.array(embedding_values))
    fig = go.Figure()
    for i in range(k_means_clusters):
        points = x_3d[kmeans.labels_ == i]
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers+text",
                text=[gene for gene, _, _, _ in papers],
                marker={"size": 6, "line": {"width": 0.5}, "opacity": 0.8},
            )
        )
    fig.update_layout(
        scene={"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "PC3"},
        width=700,
        margin={"r": 20, "b": 10, "l": 10, "t": 10},
    )

    # Save 3D plot to HTML file
    html_file_path = os.path.join(args.outdir, f"paper_clusters_{pos_or_neg}.html")
    py.write_html(fig, html_file_path)


async def cluster_papers(gene_pmid_title_abstract_dict, k_means_clusters, pos_or_neg):
    """Cluster the papers based on the embeddings of their titles and abstracts."""
    # Gather papers and their embeddings.
    papers, _, embedding_values = await get_papers_and_embeddings(gene_pmid_title_abstract_dict)
    embedding_values_array = np.array(embedding_values)

    # Propose the optimal number of clusters
    optimal_k_elbow, ssd = determine_optimal_k_elbow(embedding_values)
    plot_elbow(ssd, optimal_k_elbow, pos_or_neg, args.outdir)
    determine_optimal_k_silhouette(embedding_values_array)
    determine_optimal_k_bic(embedding_values_array)

    # Ask the user to determine the correct number of clusters, based on seeing those results
    k_means_clusters = int(input("Please enter the number of clusters: "))

    # Save thoes clusters
    clusters, kmeans = save_clusters_to_dict(embedding_values, papers, k_means_clusters)

    # With those clusters in place, visualize them in 3D
    visualize_clusters(embedding_values, papers, clusters, k_means_clusters, kmeans, pos_or_neg)

    return clusters


def save_clusters(outdir, clusters, pos_or_neg):
    """Save the clusters in a tsv of gene, pmid, cluster."""
    with open(os.path.join(outdir, f"paper_clusters_{pos_or_neg}.tsv"), "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["gene", "pmid", "cluster"])
        for cluster, papers in clusters.items():
            for gene, pmid in papers:
                writer.writerow([gene, pmid, cluster])


def sample_save_examples(seed, outdir, clusters, gene_pmid_title_abstract_dict, out_name1):
    """Randomly choose 2 papers from each cluster and ensure that gene has not been sampled.

    Then save the pmid, title, and abstract to a file.
    """
    out_name2 = out_name1.replace("examples", "examples_bkup")
    random.seed(seed)
    with open(os.path.join(outdir, out_name1), "w") as f1, open(os.path.join(outdir, out_name2), "w") as f2:
        sampled_genes = set()  # keep track of genes that have been sampled
        clustered_papers = list(clusters.items())  # convert dict to list for indexing
        i = 0  # start from the first cluster
        iterations = 0  # count the number of iterations
        while i < len(clustered_papers):
            if iterations > len(clustered_papers):  # if the number of iterations exceeds the number of clusters
                break  # break the loop
            cluster, papers = clustered_papers[i]

            random.seed(seed)
            random.shuffle(papers)  # shuffle the papers
            sampled_papers = 0  # count the number of papers sampled from this cluster
            for j, (gene, pmid) in enumerate(papers):
                title = gene_pmid_title_abstract_dict[gene][pmid]["title"]
                abstract = gene_pmid_title_abstract_dict[gene][pmid]["abstract"]
                if (
                    gene not in sampled_genes and title is not None and abstract is not None
                ):  # if the gene has not been sampled and the paper has a title and abstract
                    if sampled_papers == 0:  # if it's the first paper from this cluster
                        f = f1
                    else:  # if it's the second paper from this cluster
                        f = f2
                    f.write(f"Gene: {gene}, PMID: {pmid}\n")
                    f.write(f"Title: {title}\n")
                    f.write(f"Abstract: {abstract}\n\n")
                    sampled_genes.add(gene)  # mark the gene as sampled
                    sampled_papers += 1  # increment the number of papers sampled
                    if sampled_papers >= 2:  # if we have sampled 2 papers from this cluster
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


async def main(args):

    # Ensure that the output directory is created (and overwritten if it already exists)
    if os.path.isfile(args.outdir):
        os.remove(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    # If the user wants to fetch the paper titles and abstracts from the truth data file. This is used to sample the
    # positive few shot examples.
    if args.fetch_paper_titles_abstracts:
        # Get ground truth data (gene and associated pmids into a dictionary) from
        gene_pmid_dict = parse_tsv(args.truth_file)

        # Get and then save the title and abstract for these pmids
        pos_gene_pmid_title_abstract = get_paper_titles_abstracts(gene_pmid_dict)

        with open(args.json_file_name, "w") as f:
            json.dump(pos_gene_pmid_title_abstract, f)

        with open(args.pickle_file_name, "wb") as f:
            pickle.dump(pos_gene_pmid_title_abstract, f)

    else:
        print("\nReading the truth data pickle file: ", args.pickle_file_name)

        with open(args.pickle_file_name, "rb") as f:
            pos_gene_pmid_title_abstract = pickle.load(f)

    # Copy over the truth data files (.json and .pkl format) with the subsequent few shot results
    shutil.copy(args.json_file_name, args.outdir)
    shutil.copy(args.pickle_file_name, args.outdir)

    # Find positive examples
    if not args.just_find_negatives:
        print("\nProcessing truth data file to extract k positive examples...")

        # Cluster (based on k) the positive example papers
        clusters = await cluster_papers(pos_gene_pmid_title_abstract, args.k_means_clusters, "pos")

        # Save the clusters in a tsv of gene, pmid, cluster
        save_clusters(args.outdir, clusters, "pos")

        # Sample and save positive few shot examples
        sample_save_examples(
            args.seed, args.outdir, clusters, pos_gene_pmid_title_abstract, "few_shot_pos_examples.txt"
        )

    # Stop if only finding positives
    if args.just_find_positives:
        return

    # Proess negative examples. Note that the script will error if an associated benchmark file is not provided. The
    # benchmark file is needed to process irrelevant papers, from which we will sample our negative few shot examples.
    if args.benchmark_file:
        print("\nProcessing benchmark file to extract k negative examples...")

        # Collect negative (irrelevant) examples based on a pipeline run
        neg_dict = collect_neg_papers(args.benchmark_file)

        # Add the abstracts to those negative examples
        neg_dict = add_abstract(neg_dict)

        # Cluster (based on k) the negative example papers
        clusters = await cluster_papers(neg_dict, args.k_means_clusters, "neg")

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
        "-l",  # because pickles look like "l"s, and because I want to use "p" for positive.
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
        "-p",  # because pickles look like "l"s, and because I want to use "p" for positive.
        "--just-find-positives",
        nargs="?",
        default=False,
        type=bool,
        help="Default is False. If True, only positive examples will be identified based on a benchmark file.",
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
    )  # e.g. /home/azureuser/ev-agg-exp/.out/binary_classes_paper_finding_results_2024-04-24/
    #         benchmarking_paper_finding_results_train.txt
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

    # Throw error if just_find_negatives and just_find_positives are both True, only one can be true at a time, if any
    if args.just_find_negatives and args.just_find_positives:
        parser.error(
            "Only one of -p/--just-find-positives or -n/--just-find-negatives can be True at a time. "
            "If neither are true, both positive and negative few shot examples will be generated."
        )

    # File to determine the negative few shot examples is required when just_find_negatives is True
    if args.just_find_negatives and not args.benchmark_file:
        parser.error("-b/--benchmark-file is required when -n/--just-find-negatives is provided.")

    print("Evidence Aggregator Few Shot Example Generator is running with the following parameters:")
    for arg, value in vars(args).items():
        print(f"- {arg}: {value}")

    asyncio.run(main(args))
