"""This script compares two evagg output tables, specifically focusing on the content extraction performance.

Content extraction has two logical components:
1. Identifying the observations (Tuple[variant, individual]) in the paper.
2. Extracting the content associated with those observations.

This notebook compares the performance of the two components separately.
"""

# %% Imports.

import json
import os
import re
from collections import defaultdict
from typing import Any, List

import pandas as pd
from pyhpo import Ontology

from lib.evagg.content import HGVSVariantFactory
from lib.evagg.ref import MutalyzerClient, NcbiLookupClient, NcbiReferenceLookupClient, PyHPOClient
from lib.evagg.utils import CosmosCachingWebClient, get_dotenv_settings

# %% Constants.

TRUTH_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "v1", "evidence_train_v1.tsv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", ".out", "content_benchmark.tsv")

# TODO: after we rethink variant nomenclature, figure out whether we need to check the hgvs nomenclatures for agreement.
# alternatively set CONTENT_COLUMNS to set()  # when CONTENT_COLUMNS is empty we're just comparing observation-finding
# CONTENT_COLUMNS = {"phenotype", "zygosity", "variant_inheritance", "variant_type", "functional_study"}
CONTENT_COLUMNS = {"phenotype, variant_inheritance, zygosity"}
INDEX_COLUMNS = {"individual_id", "hgvs_c", "hgvs_p", "paper_id"}
EXTRA_COLUMNS = {"gene", "in_supplement"}

# TODO, just get the gene list from the yaml?
RESTRICT_TRUTH_GENES_TO_OUTPUT = False  # if True, only compare the genes in the output set to the truth set.

HPO_SIMILARITY_THRESHOLD = (
    0.2  # The threshold for considering two HPO terms to be the same. See sandbox/miah/hpo_pg.py.
)

# %% Read in the truth and output tables.

truth_df = pd.read_csv(TRUTH_PATH, sep="\t")
if "doi" in truth_df.columns:
    print("Warning: converting doi to paper_id")
    truth_df.rename(columns={"doi": "paper_id"}, inplace=True)

output_df = pd.read_csv(OUTPUT_PATH, sep="\t", skiprows=1)

# %% Preprocess the dataframes.


# Preprocessing of the individual ID column is necessary because there's so much variety here.
# Treat anything that is the proband, proband, unknown, and patient as the same.
def _normalize_individual_id(individual_id: Any) -> str:
    if pd.isna(individual_id):
        return "inferred proband"
    individual_id = individual_id.lower()
    if individual_id in ["the proband", "proband", "the patient", "patient", "unknown", "one proband"]:
        return "inferred proband"
    return individual_id


if "individual_id" in truth_df.columns:
    truth_df["individual_id_orig"] = truth_df["individual_id"]
    truth_df["individual_id"] = truth_df["individual_id"].apply(_normalize_individual_id)

if "individual_id" in output_df.columns:
    output_df["individual_id_orig"] = output_df["individual_id"]
    output_df["individual_id"] = output_df["individual_id"].apply(_normalize_individual_id)

# If the column "functional_study" is included in CONTENT_COLUMNS, we need to decode that column into the corresponding
# one-shot values to compare to the truth set.


def _decode_functional_study(value: str, encoding: str) -> str:
    items = json.loads(value)
    return "true" if encoding in items else "false"


if "functional_study" in CONTENT_COLUMNS:
    # Handle output_df.
    functional_column_map = {
        "engineered_cells": "cell line",
        "patient_cells_tissues": "patient cells",
        "animal_model": "animal model",
    }
    for column, encoding in functional_column_map.items():
        assert column in truth_df.columns, f"Column {column} not found in truth set."
        output_df[column] = output_df["functional_study"].apply(func=_decode_functional_study, encoding=encoding)

    CONTENT_COLUMNS -= {"functional_study"}
    CONTENT_COLUMNS.update(functional_column_map.keys())

    output_df = output_df.drop(columns=["functional_study"])

    # Handle truth_df.
    for column in functional_column_map.keys():
        truth_df[column] = truth_df[column].apply(lambda x: "true" if x == "x" else "false")

if "variant_type" in CONTENT_COLUMNS:
    # For both dataframes, recode "splice donor", "splice acceptor" to "splice region"
    # For both dataframes, recode "frameshift insertion", "frameshift deletion" to "frameshift"
    for df in [truth_df, output_df]:
        df["variant_type"] = df["variant_type"].apply(
            lambda x: "splice region" if x in ["splice donor", "splice acceptor"] else x
        )
        df["variant_type"] = df["variant_type"].apply(
            lambda x: "frameshift" if x in ["frameshift insertion", "frameshift deletion"] else x
        )


# %% Restrict the truth set to the genes in the output set.
if RESTRICT_TRUTH_GENES_TO_OUTPUT:
    print("Warning: restricting truth set to genes in the output set.")
    output_genes = set(output_df.gene.unique())
    truth_df = truth_df[truth_df.gene.isin(output_genes)]

# TODO: temporary, sample the both dfs so we have some missing/extra rows.
# truth_df = truth_df.sample(frac=0.9, replace=False)
# output_df = output_df.sample(frac=0.7, replace=False)

# %% Sanity check the dataframes.
all_columns = CONTENT_COLUMNS.union(INDEX_COLUMNS).union(EXTRA_COLUMNS)

missing_from_truth = CONTENT_COLUMNS.union(INDEX_COLUMNS) - set(truth_df.columns)
if missing_from_truth:
    raise ValueError(f"Truth table is missing columns: {missing_from_truth}")

missing_from_output = CONTENT_COLUMNS.union(INDEX_COLUMNS) - set(output_df.columns)
if missing_from_output:
    raise ValueError(f"Output table is missing columns: {missing_from_output}")

# Ensure that the index columns are unique.
if not truth_df.set_index(list(INDEX_COLUMNS)).index.is_unique:
    # Get a list of the non-unique indices
    non_unique_indices = truth_df[truth_df.duplicated(subset=list(INDEX_COLUMNS), keep=False)][list(INDEX_COLUMNS)]
    raise ValueError(f"Truth table has non-unique index columns: {non_unique_indices}")

if not output_df.set_index(list(INDEX_COLUMNS)).index.is_unique:
    # Ge ta list of the non-unique indices
    non_unique_indices = output_df[output_df.duplicated(subset=list(INDEX_COLUMNS), keep=False)][list(INDEX_COLUMNS)]
    raise ValueError(f"Output table has non-unique index columns: {non_unique_indices}")

# %% Normalize the HGVS representations from the truth data.

# TODO, consider normalizing truthset variants during generation of the truthset?

web_client = CosmosCachingWebClient(
    get_dotenv_settings(filter_prefix="EVAGG_CONTENT_CACHE_"), web_settings={"no_raise_codes": [422]}
)
mutalyzer_client = MutalyzerClient(web_client)
ncbi_client = NcbiLookupClient(web_client)
refseq_client = NcbiReferenceLookupClient(web_client)
variant_factory = HGVSVariantFactory(
    validator=mutalyzer_client,
    normalizer=mutalyzer_client,
    variant_lookup_client=ncbi_client,
    refseq_client=refseq_client,
)


def _convert_single_to_three(single_code: str) -> str:
    """Convert a single letter amino acid code to three letter."""
    from Bio.SeqUtils import IUPACData

    result = ""
    for c in single_code:
        if c.upper() == "X" or c == "*":
            result += "Ter"
        else:
            result += IUPACData.protein_letters_1to3[c.upper()]
    return result


def _bioc_convert(hgvs_desc: str) -> str:
    """Convert a p. using single letter to three letter."""
    import re

    result = re.match(r"p\.([A-Z])([0-9]+)([A-Z])", hgvs_desc)
    if not result:
        return hgvs_desc

    ref = result.group(1)
    pos = result.group(2)
    alt = result.group(3)

    ref = _convert_single_to_three(ref)
    alt = _convert_single_to_three(alt)

    result = f"p.{ref}{pos}{alt}"  # type: ignore
    return result  # type: ignore


def _normalize_hgvs(gene: str, transcript: Any, hgvs_desc: Any) -> str:
    if pd.isna(hgvs_desc):
        return hgvs_desc
    if pd.isna(transcript):
        transcript = None
    try:
        variant_obj = variant_factory.parse(text_desc=hgvs_desc, gene_symbol=gene, refseq=transcript)
    except Exception as e:
        print(f"Error normalizing {gene} {transcript} {hgvs_desc}: {e}")
        variant_obj = None

    if variant_obj and (variant_obj.valid or variant_obj.hgvs_desc.find("fs") != -1):
        return variant_obj.hgvs_desc
    elif hgvs_desc.startswith("p."):
        return _bioc_convert(hgvs_desc)
    return hgvs_desc


def _normalize_hgvs_c(row: pd.Series) -> str:
    return _normalize_hgvs(
        row.gene,
        row.transcript,
        row.hgvs_c,
    )


def _normalize_hgvs_p(row: pd.Series) -> str:
    return _normalize_hgvs(
        row.gene,
        row.transcript,
        row.hgvs_p,
    )


# %% Apply the normalization to the truth dataframes.

if "hgvs_c" in truth_df.columns:
    truth_df["hgvs_c_orig"] = truth_df["hgvs_c"]
    truth_df["hgvs_c"] = truth_df.apply(_normalize_hgvs_c, axis=1)

if "hgvs_p" in truth_df.columns:
    truth_df["hgvs_p_orig"] = truth_df["hgvs_p"]
    truth_df["hgvs_p"] = truth_df.apply(_normalize_hgvs_p, axis=1)

# %% Apply the normalization to the output dataframe.

# No need to normalize hgvs_c in output, since it's already normalized by the pipeline.
# hgvs_p on the other hand should be normalized because

# TODO: this is a hack, we should have fallback normalization in the pipeline and that truth set should be normalized
# during generation.

if "hgvs_p" in output_df.columns:
    output_df["hgvs_p_orig"] = output_df["hgvs_p"]
    output_df["hgvs_p"] = output_df.apply(_normalize_hgvs_p, axis=1)


# %% Consolidate the indices.

if "hgvs_c" in INDEX_COLUMNS and "hgvs_p" in INDEX_COLUMNS:
    print("Refactoring index columns to use hgvs_desc.")

    # Some variants are described with hgvs_c (splicing), some with hgvs_p (protein variant), and some can have both.
    # We want to consolidate these into a single index column, with preference to hgvs_c if it's not nan.
    truth_df["hgvs_desc"] = truth_df["hgvs_c"].fillna(truth_df["hgvs_p"])
    output_df["hgvs_desc"] = output_df["hgvs_c"].fillna(output_df["hgvs_p"])

    # remove the original hgvs colummns from INDEX_COLUMNS and add the new index
    INDEX_COLUMNS -= {"hgvs_c", "hgvs_p"}
    INDEX_COLUMNS.add("hgvs_desc")

    EXTRA_COLUMNS.update({"hgvs_c", "hgvs_p"})

    # Reset all_columns
    all_columns = CONTENT_COLUMNS.union(INDEX_COLUMNS).union(EXTRA_COLUMNS)

# %% Before merging, we want to Consolidate near-misses in the individual ID column on a per paper/variant basis.

for group, truth_group_df in truth_df.groupby(["paper_id", "hgvs_desc"]):
    # make a dict keyed on individual_id, with values that are the consolidated individual_ids.
    individual_id_map: dict[str, List[str]] = defaultdict(list)

    output_group_df = output_df[(output_df["paper_id"] == group[0]) & (output_df["hgvs_desc"] == group[1])]

    for individual_id in set(truth_group_df.individual_id.unique()).union(output_group_df.individual_id.unique()):
        found = False
        for key in individual_id_map:
            if individual_id == key:
                found = True
                break
            elif any(individual_id == token.lstrip("(").rstrip(")") for token in key.split()):
                values = individual_id_map.pop(key)
                values.append(individual_id)
                individual_id_map[individual_id] = values
                found = True
                break
            elif any(key == token.lstrip("(").rstrip(")") for token in individual_id.split()):
                individual_id_map[key].append(individual_id)
                found = True
                break

        if not found:
            individual_id_map[individual_id] = [individual_id]

    if any(len(v) > 1 for v in individual_id_map.values()):
        # invert the map
        mapping = {v: k for k, values in individual_id_map.items() for v in values}

        truth_df.loc[(truth_df["paper_id"] == group[0]) & (truth_df["hgvs_desc"] == group[1]), "individual_id"] = (
            truth_group_df.individual_id.map(mapping)
        )
        output_df.loc[(output_df["paper_id"] == group[0]) & (output_df["hgvs_desc"] == group[1]), "individual_id"] = (
            output_df.individual_id.map(mapping)
        )

# %% Merge the dataframes.
columns_of_interest = list(all_columns)

truth_df = truth_df.reset_index()
truth_df = truth_df.reset_index()[[c for c in columns_of_interest if c in truth_df.columns]]

output_df = output_df.reset_index()
output_df = output_df[[c for c in columns_of_interest if c in output_df.columns]]


# Add a column for provenance.
truth_df["in_truth"] = True
truth_df["in_truth"] = truth_df["in_truth"].astype("boolean")  # support nullable.
output_df["in_output"] = True
output_df["in_output"] = output_df["in_output"].astype("boolean")  # support nullable.

# reindex the two dataframes based on a multi_index from INDEX_COLUMNS.
truth_df.set_index(list(INDEX_COLUMNS), inplace=True)
output_df.set_index(list(INDEX_COLUMNS), inplace=True)

# Merge the two dataframes.
merged_df = pd.merge(
    truth_df, output_df, how="outer", left_index=True, right_index=True, suffixes=["_truth", "_output"]
)

merged_df["in_truth"] = merged_df["in_truth"].fillna(False)
merged_df["in_output"] = merged_df["in_output"].fillna(False)

if "gene_truth" in merged_df.columns:
    merged_df["gene"] = merged_df["gene_truth"].fillna(merged_df["gene_output"])
    merged_df.drop(columns=["gene_truth", "gene_output"], inplace=True)

# Reorder columns, keeping in_truth and in_output as the last two.
merged_df = merged_df[[c for c in merged_df.columns if c not in {"in_truth", "in_output"}] + ["in_truth", "in_output"]]

# %% Assess observation finding.

precision = merged_df.in_truth[merged_df.in_output == True].mean()
recall = merged_df.in_output[merged_df.in_truth == True].mean()

# Make a copy of merged_df removing all rows where in_supplement is 'Y'
merged_df_no_supplement = merged_df[merged_df.in_supplement != "Y"]
precision_no_supplement = merged_df_no_supplement.in_truth[merged_df_no_supplement.in_output == True].mean()
recall_no_supplement = merged_df_no_supplement.in_output[merged_df_no_supplement.in_truth == True].mean()

print("---- Observation finding performance ----")
print("Overall")
print(f"  Observation finding precision: {precision:.2f}")
print(f"  Observation finding recall: {recall:.2f}")
print("Ignoring truth papers from supplement")
print(f"  Observation finding precision: {precision_no_supplement:.2f}")
print(f"  Observation finding recall: {recall_no_supplement:.2f}")
print()

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

if precision < 1 or recall < 1:
    printable_df = merged_df.reset_index()  #
    printable_df[printable_df.in_supplement != "Y"].sort_values(["gene", "paper_id", "hgvs_desc"])
else:
    print("All observations found. This is likely because the Truthset observation finder was used.")
    # printable_df[
    #     (printable_df.in_supplement != "Y") & ((printable_df.in_truth != True) | (printable_df.in_output != True))
    # ].sort_values(["gene", "paper_id", "hgvs_desc"])

# %% Redo the merge and assess variant finding.

# %% Assess content extraction.

hpo = PyHPOClient()
from typing import Set, Tuple

# def _fuzzy_match_hpo_sets(left_set: str, right_set: str) -> Tuple[list[str], list[str], list[str], list[str]]:
#     # First, if both sets are nan, 'unknown', "Unknown" or empty, we'll consider them a match.
#     def _is_unknown(hpo_set: str) -> bool:
#         return pd.isna(hpo_set) or hpo_set.lower() == '["unknown"]' or hpo_set == "[]"

#     left_terms = re.findall(r"HP:\d+", left_set) if isinstance(left_set, str) and not _is_unknown(left_set) else []
#     right_terms = re.findall(r"HP:\d+", right_set) if isinstance(right_set, str) and not _is_unknown(right_set) else []

#     left_result = hpo.compare_set(left_terms, right_terms)
#     right_result = hpo.compare_set(right_terms, left_terms)

#     left_matched = [f"{k}<>{v[1]}" for k, v in left_result.items() if v[0] >= HPO_SIMILARITY_THRESHOLD]
#     right_matched = [f"{k}<>{v[1]}" for k, v in right_result.items() if v[0] >= HPO_SIMILARITY_THRESHOLD]

#     left_missed = [k for k, v in left_result.items() if v[0] < HPO_SIMILARITY_THRESHOLD]
#     right_missed = [k for k, v in right_result.items() if v[0] < HPO_SIMILARITY_THRESHOLD]

#     # Return the shared terms, the terms in left not present in right, the terms in right not present in left
#     return (left_matched, right_matched, left_missed, right_missed)


def _hpo_str_to_set(hpo_compound_string: str) -> Set[str]:
    """Takes a string of the form "Foo (HP:1234), Bar (HP:4321) and provides a set of strings that correspond to the
    HPO IDs embedded in the string.
    """
    return set(re.findall(r"HP:\d+", hpo_compound_string)) if pd.isna(hpo_compound_string) is False else set()


Ontology()
ROOT = Ontology.get_hpo_object("HP:0000001")


def _generalize_hpo_term(hpo_term: str, depth: int = 3) -> str:
    """Take an HPO term ID and return the generalized version of that term at `depth`.

    `depth` determines the degree to which hpo_term gets generalized, setting depth=1 will always return HP:0000001.

    If the provided term is more generalized than depth (e.g., "HP:0000118"), then that term itself will be returned.
    If the provided term doesn't exist in the ontology, then an error will be raised.
    """
    try:
        hpo_obj = Ontology.get_hpo_object(hpo_term)
    except RuntimeError:
        # HPO term not found in pyhpo, can't use
        print("Warning: HPO term not found in pyhpo, can't use", hpo_term)
        return ""

    try:
        path_len, path, _, _ = ROOT.path_to_other(hpo_obj)
    except RuntimeError:
        # No root found, occurs for obsolete terms.
        return hpo_obj.__str__()
    if path_len < depth:
        return hpo_obj.__str__()
    return path[depth - 1].__str__()


def _match_hpo_sets(hpo_left, hpo_right) -> Tuple[list[str], list[str], list[str], list[str]]:
    #     # First, if both sets are nan, 'unknown', "Unknown" or empty, we'll consider them a match.
    #     def _is_unknown(hpo_set: str) -> bool:
    #         return pd.isna(hpo_set) or hpo_set.lower() == '["unknown"]' or hpo_set == "[]"

    left_terms = _hpo_str_to_set(hpo_left)
    right_terms = _hpo_str_to_set(hpo_right)

    left_gen = {_generalize_hpo_term(t) for t in left_terms}
    left_gen -= {""}  # Remove empty strings.
    right_gen = {_generalize_hpo_term(t) for t in right_terms}
    right_gen -= {""}  # Remove empty strings.

    matches = list(left_gen.intersection(right_gen))
    return matches, matches, list(left_gen - right_gen), list(right_gen - left_gen)


# %%

if CONTENT_COLUMNS:
    shared_df = merged_df_no_supplement[merged_df_no_supplement.in_truth & merged_df_no_supplement.in_output]

    print("---- Content extraction performance ----")

    for column in CONTENT_COLUMNS:
        if column == "phenotype":
            # Phenotype matching can't be a direct string compare.
            # We'll fuzzy match the HPO terms, note that we're ignoring anything in here that couldn't be matched via
            # HPO. The non HPO terms are still provided by pipeline output, but
            pheno_stats = shared_df.apply(
                lambda row: _match_hpo_sets(row["phenotype_truth"], row["phenotype_output"]), axis=1
            )
            match = pheno_stats.apply(lambda x: len(x[2]) == 0 and len(x[3]) == 0)
            print(f"Content extraction accuracy for {column}: {match.mean():.3f} (of N={match.count()})")
        else:
            # Currently other content columns are just string compares.
            match = shared_df[f"{column}_truth"].str.lower() == shared_df[f"{column}_output"].str.lower()
            print(f"Content extraction accuracy for {column}: {match.mean():.3f} (of N={match.count()})")

        for idx, row in shared_df.iterrows():
            if match[idx]:  # type: ignore
                # print(f"!!Match ({idx}): {row[f'{column}_truth']} == {row[f'{column}_output']}")
                pass
            else:
                # print(f"  Mismatch ({idx}): {row[f'{column}_truth']} != {row[f'{column}_output']}")
                print(f"##Mismatch ({idx})")
                for i, x in enumerate(pheno_stats[idx]):
                    if i != 0:
                        print(f"  {x}")
                print(f"  Truth: {row[f'{column}_truth']}")
                print(f"  Output: {row[f'{column}_output']}")
                print()
                pass
        print()

# %%
