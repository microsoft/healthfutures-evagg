import logging
from typing import Any, Dict, Sequence

from lib.evagg.types import Paper

logger = logging.getLogger(__name__)


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
