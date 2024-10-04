"""This notebook is used to generate text questions for the discrepancy resolution task.

This notebook can only be run after paper_finding/manuscript_discrepancy_list.py,
content_extraction/manuscript_obs_discrepancy_list.py, and content_extraction/manuscript_content_discrepancy_list.py
have been run. As it requires outputs of those scripts as inputs.
"""

# %% Imports.

import os
from functools import cache
from typing import Any, Dict, List

import pandas as pd

from lib.di import DiContainer
from lib.evagg.ref import IFetchHPO
from scripts.benchmark.utils import INDICES_FOR_COLUMN, get_paper

# %% Constants.

OUTPUT_DIR = ".out/manuscript_content_extraction"

PAPER_FINDING_DISCREPANCY_FILE = ".out/manuscript_paper_finding/all_papers_list.tsv"

OBSERVATION_DISCREPANCY_FILE = ".out/manuscript_content_extraction/all_observations_list.tsv"

CONTENT_DISCREPANCY_ROOT = ".out/manuscript_content_extraction"

MAX_QUESTIONS_PER_FILE = 40

# Note here we have excluded animal_model, engineered_cells, and patient_cells_tissues. We could opt to include
# them later if we think we have time from the analysts to review. Also note we are including papers and observations
# here so that we can generate questions for them as well.

INDICES_FOR_COLUMN_RELEVANT_TASKS = INDICES_FOR_COLUMN
INDICES_FOR_COLUMN_RELEVANT_TASKS["papers"] = ["gene", "pmid"]
INDICES_FOR_COLUMN_RELEVANT_TASKS["observations"] = ["gene", "pmid", "hgvs_desc", "individual_id"]
INDICES_FOR_COLUMN_RELEVANT_TASKS.pop("animal_model")
INDICES_FOR_COLUMN_RELEVANT_TASKS.pop("engineered_cells")
INDICES_FOR_COLUMN_RELEVANT_TASKS.pop("patient_cells_tissues")

# %% Functions.


@cache
def _get_hpo_reference() -> IFetchHPO:
    hpo_reference = DiContainer().create_instance({"di_factory": "lib.evagg.ref.PyHPOClient"}, {})
    return hpo_reference


def gen_header(current: int, total: int):
    return f"Discrepancy Questionnaire ({current} of {total})\n"


def gen_group_header(gene: str, pmid: str) -> str:
    paper = get_paper(pmid)

    if paper is None:
        title = "Unknown title"
        link = "Unknown link"
    else:
        title = paper.props.get("title", "Unknown title")
        if pmcid := paper.props.get("pmcid"):
            link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
        else:
            link = paper.props.get("link", "Unknown link")

    return f'The following section contains questions pertaining to the gene {gene} and paper "{title}" ({link}).\n'


def gen_paper_finding_question(idx: str, gene: str, pmid: str) -> str:
    return (
        f"\tQ{idx}. The paper {pmid} discusses one or more human genetic variants in the gene {gene}. "
        "Select one response.\n"
        "\t\tTrue\n"
        "\t\tFalse\n"
        "\t\t[Optional] Note\n"
    )


def gen_obs_question(idx: str, gene: str, pmid: str, hgvs_desc: str, individual_id: str) -> str:
    if individual_id == "inferred proband":
        question_text = (
            f"\tQ{idx}. The paper {pmid} "
            f"discusses the variant {hgvs_desc} in {gene} in the primary proband or an unknown/unnamed individual. "
            "Select one response.\n"
            "\t\tThis is completely correct\n"
            f"\t\tThis variant is discussed, but it is not associated with the gene {gene}\n"
            "\t\tThis variant is discussed, but it is not possessed by the primary proband "
            "or an unknown/unnamed individual\n"
            "\t\tThis variant is either invalid or it is not discussed in the paper\n"
            "\t\t[Optional] Note\n"
        )
    else:
        question_text = (
            f"\tQ{idx}. The paper {pmid} "
            f'discusses the variant {hgvs_desc} in {gene} possessed by the individual "{individual_id}". '
            "Select one response. \n"
            "\t\tThis is completely correct\n"
            f"\t\tThis variant is discussed, but it is not associated with the gene {gene}\n"
            f'\t\tThis variant is discussed, but it is not possessed by the individual "{individual_id}"\n'
            "\t\tThis variant is either invalid or it is not discussed in the paper.\n"
            "\t\t[Optional] Note\n"
        )
    return question_text


def _model_question(idx: str, col: str, pmid: str, hgvs_desc: str) -> str:
    return (
        f"\tQ{idx}. The paper {pmid} "
        f"discusses the functional data from '{col.replace('_', ' ')}' for the variant {hgvs_desc}. "
        "Select one response. \n"
        "\t\tTrue, the paper discusses functional data from this source\n"
        "\t\tFalse, the paper does not discuss functional data from this source\n"
        "\t\t[Optional] Note\n"
    )


def _phenotype_question(
    idx: str,
    pmid: str,
    hgvs_desc: str,
    individual_id: str,
    output_dict: Dict[str, List[str]],
    truth_dict: Dict[str, List[str]],
) -> str:
    if individual_id == "inferred proband":
        question_text = (
            f"\tQ{idx}. The paper {pmid} "
            f"discusses the phenotype for the primary proband or an unknown/unnamed individual with "
            f"variant '{hgvs_desc}'. Select all phenotypes possessed by the individual:\n"
        )
    else:
        question_text = (
            f"\tQ{idx}. The paper {pmid} "
            f"discusses the phenotype for the individual '{individual_id}' with variant '{hgvs_desc}'. "
            "Select all phenotypes possessed by the individual:\n"
        )
    spec_terms = set()
    for v in output_dict.values():
        for t in v:
            spec_terms.add(t)
    for v in truth_dict.values():
        for t in v:
            spec_terms.add(t)
    if not spec_terms:
        raise ValueError("No HPO terms found for phenotype discrepancy.")
    hpo_reference = _get_hpo_reference()
    for t in spec_terms:
        if (hpo_term := hpo_reference.fetch(t)) is not None:
            question_text += f"\t\t{hpo_term['name']} ({hpo_term['id']})\n"
    question_text += "\t\t[Optional] Note\n"
    return question_text


def _study_type_question(idx: str, pmid: str) -> str:
    question_text = f"\tQ{idx}. What type of study is the paper {pmid}? Select one response.\n"
    for study_type in ["Case report", "Case series", "Cohort analysis", "Other", "[Optional] Note"]:
        question_text += f"\t\t{study_type}\n"
    return question_text


def _variant_inheritance_question(idx: str, pmid: str, hgvs_desc: str, individual_id: str) -> str:
    if individual_id == "inferred proband":
        question_text = (
            f"\tQ{idx}. The paper {pmid} "
            f"discusses the inheritance pattern for the variant '{hgvs_desc}' possessed by the primary proband or an "
            f"unknown individual. What is the actual inheritance pattern in this case? Select one response.\n"
        )
    else:
        question_text = (
            f"\tQ{idx}. The paper {pmid} "
            f"discusses the inheritance pattern for the variant '{hgvs_desc}' possessed by the individual "
            f"'{individual_id}'. What is the actual inheritance pattern in this case? Select one response.\n"
        )
    for value in ["De novo", "Inherited", "Unknown", "[Optional] Note"]:
        question_text += f"\t\t{value}\n"
    return question_text


def _variant_type_question(idx: str, pmid: str, hgvs_desc: str) -> str:
    potential_values = (
        [
            "Missense",
            "Stop gained",
            "Stop lost",
            "Splice region",
            "Frameshift",
            "Synonymous",
            "Inframe deletion",
            "Indel",
            "Unknown",
            "[Optional] Note",
        ],
    )
    question_text = (
        f"\tQ{idx}. The paper {pmid} "
        f"discusses the variant type for {hgvs_desc}. What is the actual variant type in this case? "
        "Select one response.\n"
    )
    for value in potential_values:
        question_text += f"\t\t{value}\n"
    return question_text


def _zygosity_question(idx: str, pmid: str, hgvs_desc: str, individual_id: str) -> str:
    potential_values = ["Homozygous", "Heterozygous", "Compound heterozygous", "Unknown", "[Optional] Note"]

    if individual_id == "inferred proband":
        question_text = (
            f"\tQ{idx}. The paper {pmid} "
            f"discusses the zygosity for the variant '{hgvs_desc}' possessed by the primary proband or an unknown "
            "individual. What is the actual zygosity in this case? Select one response.\n"
        )
    else:
        question_text = (
            f"\tQ{idx}. The paper {pmid} "
            f"discusses the zygosity for the variant '{hgvs_desc}' possessed by the individual '{individual_id}'. "
            "What is the actual zygosity in this case? Select one response.\n"
        )
    for value in potential_values:
        question_text += f"\t\t{value}\n"
    return question_text


def gen_content_question(idx: str, col: str, params: Dict[str, Any]) -> str:
    if col in ["animal_model", "engineered_cells", "patient_cells_tissues"]:
        return _model_question(idx, col, params["pmid"], params["hgvs_desc"])
    elif col in ["phenotype"]:
        return _phenotype_question(
            idx,
            params["pmid"],
            params["hgvs_desc"],
            params["individual_id"],
            params["output_dict"],
            params["truth_dict"],
        )
    elif col in ["study_type"]:
        return _study_type_question(idx, params["pmid"])
    elif col in ["variant_inheritance", "zygosity"]:
        return _variant_inheritance_question(idx, params["pmid"], params["hgvs_desc"], params["individual_id"])
    elif col in ["variant_type"]:
        return _variant_type_question(idx, params["pmid"], params["hgvs_desc"])
    elif col in ["zygosity"]:
        return _zygosity_question(idx, params["pmid"], params["hgvs_desc"], params["individual_id"])
    else:
        raise ValueError(f"Unknown column: {col}")


# %% Load in the souce data. It should all go into a big dataframe with two columns, pmid and question.

all_discrepancies: List[pd.DataFrame] = []

for column in INDICES_FOR_COLUMN_RELEVANT_TASKS.keys():
    if column == "papers":
        root = ".out/manuscript_paper_finding"
    else:
        root = CONTENT_DISCREPANCY_ROOT

    discrepancy_file = f"all_{column}_list.tsv"
    col_discrepancies = pd.read_csv(os.path.join(root, discrepancy_file), sep="\t")

    indices_to_keep = INDICES_FOR_COLUMN_RELEVANT_TASKS[column]
    if column == "phenotype":
        indices_to_keep += ["truth_dict", "output_dict"]
        col_discrepancies["truth_dict"] = col_discrepancies["truth_dict"].apply(eval)
        col_discrepancies["output_dict"] = col_discrepancies["output_dict"].apply(eval)

    col_discrepancies = col_discrepancies.query("discrepancy == True")[indices_to_keep]
    col_discrepancies["task"] = column
    # Make a new column called "id" that contains a tuple of the values in gene and pmid
    col_discrepancies["id"] = col_discrepancies.apply(
        lambda x, col=column: tuple(x[INDICES_FOR_COLUMN_RELEVANT_TASKS[col]]), axis=1
    )
    all_discrepancies.append(col_discrepancies)

discrepancies = pd.concat(all_discrepancies).reset_index()

# Order by pmid
discrepancies = discrepancies.sort_values(by=["gene", "pmid"])

# Now the index is a a valid question number, so name the index "question_number"
discrepancies.index.name = "question_number"

# Sort the columns so that the order is pmid, gene, hgvs_desc, individual_id, task, id, then anything else.
ordered_cols = ["pmid", "gene", "hgvs_desc", "individual_id", "task", "id"]
other_cols = [col for col in discrepancies.columns if col not in ordered_cols]
discrepancies = discrepancies[ordered_cols + other_cols]

# %% Write out the consolidated discrepancies as a dataframe for future use.

discrepancies.to_csv(os.path.join(OUTPUT_DIR, "all_sorted_discrepancies.tsv"), sep="\t")

# %% Write out the consolidated discrepancies as a series of text files for the analysts to review.

# Write out the consolidated discrepancies as a series of text files for the analysts to review.
min_questions_per_file = 40
current_file = 0
n_qs_assigned = 0

discrepancies["file_number"] = -1

# Determine which file each question is going to go in.
for _, group_df in discrepancies.groupby(["gene", "pmid"]):
    n_new_qs = len(group_df)
    n_qs_assigned += n_new_qs

    # assign the current file number to the group
    discrepancies.loc[group_df.index, "file_number"] = current_file

    if n_qs_assigned >= min_questions_per_file:
        current_file += 1
        n_qs_assigned = 0

# Now write out the questions to files.

n_files = discrepancies["file_number"].nunique()

for file_number, group_df in discrepancies.groupby("file_number"):
    # Not the cleanest way to do this, but it works.
    file_number_int = int(group_df["file_number"].iloc[0])

    with open(os.path.join(OUTPUT_DIR, f"discrepancies_qs_{file_number}.tsv"), "w") as f:
        f.write(f"{gen_header(file_number_int + 1, n_files)}\n")

        for (gene, pmid), paper_df in group_df.groupby(["gene", "pmid"]):
            f.write(f"{gen_group_header(gene, pmid)}\n\n")
            for idx, row in paper_df.iterrows():
                idx_str = str(idx)
                if row["task"] == "papers":
                    q = gen_paper_finding_question(idx_str, row["gene"], row["pmid"])
                elif row["task"] == "observations":
                    q = gen_obs_question(idx_str, row["gene"], row["pmid"], row["hgvs_desc"], row["individual_id"])
                else:
                    q = gen_content_question(idx_str, row["task"], row.to_dict())
                f.write(f"{q}\n")

# %% Intentionally empty.
