"""This notebook collects a list of all observation discrepancies between run outputs and the truth data."""

# %% Imports.

import os
from functools import cache
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lib.di import DiContainer
from lib.evagg.ref import IPaperLookupClient
from lib.evagg.types import Paper

# %% Constants.

TRAIN_RUNS = [
    "20240909_165847",
    "20240909_210652",
    "20240910_044027",
    "20240910_134659",
    "20240910_191020",
]

TEST_RUNS = [
    "20240911_165451",
    "20240911_194240",
    "20240911_223218",
    "20240912_145606",
    "20240912_181121",
]

COLUMNS_OF_INTEREST = [
    "animal_model",
    "engineered_cells",
    "patient_cells_tissues",
    "phenotype",
    "study_type",
    "variant_inheritance",
    "variant_type",
    "zygosity",
]

POTENTIAL_VALUES = {
    "animal_model": [True, False],
    "engineered_cells": [True, False],
    "patient_cells_tissues": [True, False],
    "phenotype": [],
    "study_type": ["case report", "case series", "cohort analysis", "other"],
    "variant_inheritance": ["de novo", "inherited", "unknown"],
    "variant_type": [
        "missense",
        "stop gained",
        "stop lost",
        "splice region",
        "frameshift",
        "synonymous",
        "inframe deletion",
        "indel",
        "unknown",
    ],
    "zygosity": ["homozygous", "heterozygous", "compound heterozygous", "unknown"],
}

# The number of times a discrepancy must appear in order to be included in the output.
# By setting MIN_RECURRENCE to 3, with 5 each TRAIN and TEST runs, we're looking for discrepancies that appear in more
# than half the runs.
MIN_RECURRENCE = 3

OUTPUT_DIR = ".out/manuscript_content_extraction"

# %% Function definitions.


def load_run(run_id: str, run_filename: str) -> pd.DataFrame | None:
    """Load the data from a single run."""
    run_file = f".out/run_evagg_pipeline_{run_id}_content_extraction_benchmarks/{run_filename}"
    if not os.path.exists(run_file):
        print(
            f"No benchmark analysis exists for run_id {run_file}. "
            "Do you need to run 'manuscript_content_extraction.py' first?"
        )
        return None

    run_data = pd.read_csv(run_file, sep="\t")
    return run_data


def get_eval_df(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df.empty:
        return df

    assert df[f"{column}_result_type"].nunique() == 1

    result_type = df[f"{column}_result_type"].iloc[0]

    if result_type == "I":
        eval_df = df[~df.reset_index().set_index(["paper_id", "individual_id"]).index.duplicated(keep="first")]
        indices = ["gene", "paper_id", "individual_id"]
    elif result_type == "IV":
        eval_df = df[
            ~df.reset_index().set_index(["paper_id", "hgvs_desc", "individual_id"]).index.duplicated(keep="first")
        ]
        indices = ["gene", "paper_id", "hgvs_desc", "individual_id"]
    elif result_type == "P":
        eval_df = df[~df.reset_index().set_index(["paper_id"]).index.duplicated(keep="first")]
        indices = ["gene", "paper_id"]
    elif result_type == "V":
        eval_df = df[~df.reset_index().set_index(["paper_id", "hgvs_desc"]).index.duplicated(keep="first")]
        indices = ["gene", "paper_id", "hgvs_desc"]
    else:
        raise ValueError(f"Unknown result type: {result_type}")

    return eval_df[indices + [f"{column}_result", f"{column}_truth", f"{column}_output", f"{column}_result_type"]]


@cache
def get_lookup_client() -> IPaperLookupClient:
    ncbi_lookup: IPaperLookupClient = DiContainer().create_instance({"di_factory": "lib/config/objects/ncbi.yaml"}, {})
    return ncbi_lookup


@cache
def get_paper(pmid: str) -> Paper | None:
    client = get_lookup_client()
    try:
        return client.fetch(pmid)
    except Exception as e:
        print(f"Error getting title for paper {pmid}: {e}")

    return None


# %% Generate run stats for observation finding.


def get_row_key(row: pd.Series, type_col: str) -> Tuple:
    if row[type_col] == "V":
        return row["gene"], row["paper_id"], row["hgvs_desc"]
    elif row[type_col] == "IV":
        return row["gene"], row["paper_id"], row["hgvs_desc"], row["individual_id"]
    elif row[type_col] == "I":
        return row["gene"], row["paper_id"], row["individual_id"]
    elif row[type_col] == "P":
        return row["gene"], row["paper_id"]
    else:
        raise ValueError(f"Unknown column: {type_col}")


dicts: Dict[str, Dict[Tuple[Any, ...], Dict[str, Any]]] = {col: {} for col in COLUMNS_OF_INTEREST}

for run_type, run_ids in [("train", TRAIN_RUNS), ("test", TEST_RUNS)]:

    runs = [load_run(id, "content_extraction_results.tsv") for id in run_ids]

    for run in runs:
        if run is None:
            continue

        print(f"{run_type} - {run.paper_id.nunique()} papers")

        # strip the pmid: prefix from paper_id
        run["paper_id"] = run["paper_id"].str.replace("pmid:", "").astype(int)

        for col in COLUMNS_OF_INTEREST:
            eval_df = get_eval_df(run, col)

            for _, row in eval_df.iterrows():
                key = get_row_key(row, f"{col}_result_type")

                if col == "phenotype":
                    result = eval(str(row["phenotype_result"]))
                    if key not in dicts[col]:
                        dicts[col][key] = {
                            "truth_dict": result[2],
                            "truth_count": {k: 1 for k in result[2].keys()},
                            "output_dict": result[3],
                            "output_count": {k: 1 for k in result[3].keys()},
                            "gene_group": run_type,
                            "total_count": 1,
                        }
                    else:
                        dicts[col][key]["total_count"] += 1
                        for k in result[2]:
                            if k not in dicts[col][key]["truth_dict"]:
                                dicts[col][key]["truth_dict"][k] = result[2][k]
                                dicts[col][key]["truth_count"][k] = 1
                            else:
                                dicts[col][key]["truth_dict"][k] += result[2][k]
                                dicts[col][key]["truth_count"][k] += 1
                        for k in result[3]:
                            if k not in dicts[col][key]["output_dict"]:
                                dicts[col][key]["output_dict"][k] = result[3][k]
                                dicts[col][key]["output_count"][k] = 1
                            else:
                                dicts[col][key]["output_dict"][k] += result[3][k]
                                dicts[col][key]["output_count"][k] += 1
                else:
                    if key not in dicts[col]:
                        dicts[col][key] = {
                            "total_count": 1,
                            "agree_count": int(row[f"{col}_result"]),
                            "gene_group": run_type,
                        }
                    else:
                        dicts[col][key]["total_count"] += 1
                        if row[f"{col}_result"]:
                            dicts[col][key]["agree_count"] += 1

# Convert the dictionaries to dataframes.
dfs = {col: pd.DataFrame(dicts[col]).T for col in COLUMNS_OF_INTEREST}

# %% Annotate the dataframes

for col, df in dfs.items():
    if col == "phenotype":
        # Define an output discrepancy for a row as having any value in output_count be greater than or equal to half of the total_count for that row.
        df["output_discrepancy"] = df.apply(
            lambda row: any([row["output_count"][k] >= row["total_count"] / 2 for k in row["output_count"]]), axis=1
        )
        # Same thing for truth discrepancies.
        df["truth_discrepancy"] = df.apply(
            lambda row: any([row["truth_count"][k] >= row["total_count"] / 2 for k in row["truth_count"]]), axis=1
        )
        # If either the output or truth discrepancy is true, then the phenotype discrepancy is true.
        df["discrepancy"] = df["output_discrepancy"] | df["truth_discrepancy"]
    else:
        df["discrepancy"] = (df["total_count"] - df["agree_count"]) >= MIN_RECURRENCE
    print(f"Found {df['discrepancy'].sum()} discrepancies for {col}.")


# %% Write the dataframes to a CSV file.
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for col, df in dfs.items():
    df.to_csv(os.path.join(OUTPUT_DIR, f"all_{col}_list.csv"))

# %% Write out a text file listing all discrepancies.

RANDOM_ORDER = False
RANDOM_SEED = 1

hpo = DiContainer().create_instance({"di_factory": "lib.evagg.ref.PyHPOClient"}, {})

for col, df in dfs.items():
    discrepancies = df.query("discrepancy == True").copy()

    if RANDOM_ORDER:
        discrepancies = discrepancies.sample(frac=1, random_state=RANDOM_SEED)
    else:
        discrepancies = discrepancies.sort_index(level=1)

    with open(os.path.join(OUTPUT_DIR, f"{col}_discrepancies.txt"), "w") as f:
        count = 0
        for idx in discrepancies.index.values:
            count += 1

            # Cheating here, but we know the 2nd item in idx is the pmid
            paper = get_paper(idx[1])

            if paper is None:
                title = "Unknown title"
                link = "Unknown link"
            else:
                title = paper.props.get("title", "Unknown title")
                if pmcid := paper.props.get("pmcid"):
                    link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
                else:
                    link = paper.props.get("link", "Unknown link")

            if col in ["animal_model", "engineered_cells", "patient_cells_tissues"]:
                f.write(
                    f'{count}. The paper "{title}" ({link}) '
                    f"discusses the functional data from '{col.replace('_', ' ')}' for the variant {idx[2]}\n"
                )
                f.write("A. True, the paper discusses functional data from this source\n")
                f.write("B. False, the paper does not discuss functional data from this source\n\n\n")
            elif col in ["phenotype"]:
                f.write(
                    f'{count}. The paper "{title}" ({link}) '
                    f"discusses the phenotype for the individual '{idx[3]}' with variant '{idx[2]}'\n"
                )
                f.write("Select all phenotypes posessed by the individual:\n")
                spec_terms = set()
                for _, v in discrepancies.loc[idx].output_dict.items():
                    for t in v:
                        spec_terms.add(t)
                for _, v in discrepancies.loc[idx].truth_dict.items():
                    for t in v:
                        spec_terms.add(t)
                if not spec_terms:
                    f.write("ERROR! No specific terms identified for discrepancy\n")
                for t in spec_terms:
                    if (hpo_term := hpo.fetch(t)) is not None:
                        f.write(f"{hpo_term['name']} ({hpo_term['id']})\n")
                f.write("\n\n")
            elif col in ["study_type"]:
                f.write(f'{count}. What type of study is the paper "{title}" ({link})?\n')
                letter_prefix = "A"
                for study_type in POTENTIAL_VALUES[col]:
                    f.write(f"{letter_prefix}. {study_type}\n")
                    letter_prefix = chr(ord(letter_prefix) + 1)
                f.write("\n\n")
            elif col in ["variant_inheritance", "zygosity"]:
                if idx[3] == "inferred proband":
                    f.write(
                        f'{count}. The paper "{title}" ({link}) '
                        f"discusses the {col} for the variant {idx[2]} possessed by the primary proband or an unknown "
                        f"individual. What is the actual {col} in this case?\n"
                    )
                else:
                    f.write(
                        f'{count}. The paper "{title}" ({link}) '
                        f'discusses the {col} for the variant {idx[2]} possessed by the individual "{idx[3]}". What is '
                        f"the actual {col} in this case?\n"
                    )
                letter_prefix = "A"
                for value in POTENTIAL_VALUES[col]:
                    f.write(f"{letter_prefix}. {value}\n")
                    letter_prefix = chr(ord(letter_prefix) + 1)
                f.write("\n\n")
            elif col in ["variant_type"]:
                f.write(
                    f'{count}. The paper "{title}" ({link}) '
                    f"discusses the variant type for {idx[2]}. What is the actual variant type in this case?\n"
                )
                letter_prefix = "A"
                for value in POTENTIAL_VALUES[col]:
                    f.write(f"{letter_prefix}. {value}\n")
                    letter_prefix = chr(ord(letter_prefix) + 1)
                f.write("\n\n")
            else:
                raise ValueError(f"Unknown column: {col}")

# %%
