"""Exploratory script to demonstrate/evaluate the phenotype generalization approach."""

# %% Imports.

import re

import pandas as pd
from pyhpo import Ontology

from scripts.benchmark.utils import generalize_hpo_term

Ontology()

# %% Constants.


# %% First, let's get a list of all the terms that are children of Phenotypic abnormality.

# Phenotypic abnormality.
parent_term = Ontology.get_hpo_object("HP:0000118")

print(f"HP:0000118 | Phenotypic abnormality has {len(parent_term.children)} children:")
for c in parent_term.children:
    print(f"{c.id} | {c.name}")


# %% Now let's check to make sure every phenotype in the the truth set is indeed a child of this term.

test_df = pd.read_csv("data/v1.1/evidence_test_v1.1.tsv", sep="\t")
train_df = pd.read_csv("data/v1.1/evidence_train_v1.1.tsv", sep="\t")

truth_df = pd.concat([train_df, test_df], ignore_index=True)

# %%

for idx, row in truth_df.iterrows():
    if pd.isna(row.phenotype):
        continue
    # find all instances of HP:\d+ in row.pheontype
    hpo_terms = re.findall(r"HP:\d+", row.phenotype)
    for term in hpo_terms:
        gen_term = generalize_hpo_term(term, 2)
        if not gen_term.startswith("HP:0000118"):
            print(f"{idx}: {gen_term}")

# %%

all_terms = set()

for _, row in truth_df.iterrows():
    if pd.isna(row.phenotype):
        continue
    # find all instances of HP:\d+ in row.pheontype
    hpo_terms = re.findall(r"HP:\d+", row.phenotype)
    for term in hpo_terms:
        gen_term = generalize_hpo_term(term, 3)
        hpo_str = gen_term.split("|")[0].strip()
        all_terms.add(hpo_str)

print(f"Found {len(all_terms)} unique terms.")
for t in all_terms:
    print(t)

# %% Now lets load an example pipeline output

pip_df = pd.read_csv(".out/run_evagg_pipeline_20240909_165847/pipeline_benchmark.tsv", sep="\t", skiprows=1)
# comp_df = pd.read_csv(
#     ".out/run_evagg_pipeline_20240909_165847_content_extraction_benchmarks/content_extraction_results.tsv", sep="\t"
# )

# %% Get the distribution of L3 terms


def hpo_str_to_list(hpo_compound_string: str) -> list[str]:
    """Convert a delimited list of HPO terms to a set of terms.

    Takes a string of the form "Foo (HP:1234), Bar (HP:4321) and provides a set of strings that correspond to the
    HPO IDs embedded in the string.
    """
    return list(set(re.findall(r"HP:\d+", hpo_compound_string))) if pd.isna(hpo_compound_string) is False else []


pip_df["phenotype_list"] = pip_df.phenotype.apply(hpo_str_to_list)

# Now phenotype_list is contains lists of terms, make a new dataframe with all the unique terms across all lists.
pip_df_exploded = (
    pip_df[["evidence_id", "phenotype_list"]]
    .explode("phenotype_list")
    .dropna(subset=["phenotype_list"])
    .drop_duplicates(subset=["phenotype_list"])
    .reset_index()
)
pip_df_exploded.rename(columns={"phenotype_list": "term"}, inplace=True)

# %%
pip_df_exploded["parent_term"] = pip_df_exploded.term.apply(lambda x: generalize_hpo_term(x))

pip_df_exploded["parent_term"].value_counts()

# %%
pip_df_exploded["super_parent_term"] = pip_df_exploded.term.apply(lambda x: generalize_hpo_term(x, 2))
pip_df_exploded["super_parent_term"].value_counts()

# %% Finally, create the descendency graph for each term.

from typing import List


def get_lineage(term: str) -> List[str]:
    try:
        obj = Ontology.get_hpo_object(term)
    except RuntimeError:
        return ["Unknown"]
    steps, path, _, _ = Ontology.get_hpo_object("HP:0000001").path_to_other(obj)
    return [f"{x.id} | {x.name}" for x in path[1:]]


def print_lineage(terms: List[str], indent_step: int = 2) -> None:
    indent = " " * indent_step
    for term in terms:
        print(f"{indent}{term}")
        indent += " " * indent_step


pip_df_exploded["lineage"] = pip_df_exploded.term.apply(get_lineage)

for idx, row in pip_df_exploded.sample(10).iterrows():
    print_lineage(row.lineage)
    print()

# %% Let's play with an alternative to generalize_hpo_term

from typing import Set

from pyhpo import HPOTerm

children: Set[HPOTerm] = set()

root = Ontology.get_hpo_object("HP:0000001")
for child in root.children:
    for grandchild in child.children:
        children.add(grandchild)


# Instead of getting *the only* parent term at the target level,
# let's get all the parent terms at the target level.
def get_all_parent_terms(term: HPOTerm) -> Set[HPOTerm]:
    if term in children:
        return {term}
    return children.intersection(term.all_parents)


def get_all_parent_terms_by_id(term_id: str) -> Set[HPOTerm]:
    try:
        term = Ontology.get_hpo_object(term_id)
    except RuntimeError:
        return set()
    return get_all_parent_terms(term)


pip_df_exploded["parent_set"] = pip_df_exploded.term.apply(get_all_parent_terms_by_id)

pip_df_exploded["n_parents"] = pip_df_exploded.parent_set.apply(len)

pip_df_exploded["n_parents"].value_counts()

# %% Ask the same question of the truth set hpo terms.

truth_df["phenotype_ids"] = truth_df.phenotype.apply(hpo_str_to_list)
truth_df_exploded = truth_df.explode("phenotype_ids").dropna(subset=["phenotype_ids"]).reset_index()
truth_df_exploded.rename(columns={"phenotype_ids": "term"}, inplace=True)

truth_df_exploded["parent_set"] = truth_df_exploded.term.apply(get_all_parent_terms_by_id)
truth_df_exploded["n_parents"] = truth_df_exploded.parent_set.apply(len)
truth_df_exploded["n_parents"].value_counts()

# %%
