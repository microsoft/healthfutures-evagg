"""This script compares two evagg output tables, specifically focusing on the content extraction performance.

Content extraction has two logical components:
1. Identifying the observations (Tuple[variant, individual]) in the paper.
2. Extracting the content associated with those observations.

This notebook compares the performance of the two components separately.
"""

# %% Imports.

import argparse
import os
import re
from collections import defaultdict
from functools import cache
from typing import Any, Dict, List, Set, Tuple, Type

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Bio.SeqUtils import IUPACData
from pyhpo import Ontology
from sklearn.metrics import confusion_matrix

from lib.di import DiContainer
from lib.evagg.content import HGVSVariantFactory
from lib.evagg.utils.run import get_previous_run

# Function definitions.


def get_pipeline_output(args: argparse.Namespace) -> Tuple[str, pd.DataFrame]:

    if args.pipeline_output:
        run_record = get_previous_run("evagg_pipeline", name_includes=args.pipeline_output)
    else:
        run_record = get_previous_run("evagg_pipeline")

    if not run_record or not run_record.path or not run_record.output_file:
        raise ValueError("No pipeline output found.")

    # Read in the corresponding pipeline output. Assume we're running from repo root.
    pipeline_df = pd.read_csv(os.path.join(run_record.path, run_record.output_file), sep="\t", skiprows=1)

    # If paper_id is prefixed with "pmid:", remove it.
    pipeline_df["pmid"] = pipeline_df["paper_id"].str.lstrip("pmid:").astype(int)

    return (run_record.path, pipeline_df)


def get_mgt(args: argparse.Namespace) -> pd.DataFrame:
    truth_df = pd.read_csv(args.mgt_train_test_path, sep="\t")
    if "doi" in truth_df.columns:
        print("Warning: converting doi to paper_id")
        truth_df.rename(columns={"doi": "paper_id"}, inplace=True)

    return truth_df


def _normalize_individual_id(individual_id: Any) -> str:
    # Preprocessing of the individual ID column is necessary because there's so much variety here.
    # Treat anything that is the proband, proband, unknown, and patient as the same.
    if pd.isna(individual_id):
        return "inferred proband"
    individual_id = individual_id.lower()
    if individual_id in ["the proband", "proband", "the patient", "patient", "unknown", "one proband"]:
        return "inferred proband"
    return individual_id


def normalize_individual_ids_in_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize individual ids in a dataframe."""
    if "individual_id" in df.columns:
        df["individual_id_orig"] = df["individual_id"]
        df["individual_id"] = df["individual_id"].apply(_normalize_individual_id)
    return df


def recode_field_columns_in_df(df: pd.DataFrame, content_columns: Set[str]) -> pd.DataFrame:

    # Remap functional study columns to use string values "true" and "false" instead of boolean True and False.
    for col in ["engineered_cells", "patient_cells_tissues", "animal_model"]:
        if col in content_columns:
            df[col] = df[col].apply(lambda x: "true" if x else "false")

    if "variant_type" in content_columns:
        # Recode "splice donor", "splice acceptor" to "splice region"
        # Recode "frameshift insertion", "frameshift deletion" to "frameshift"
        df["variant_type"] = df["variant_type"].apply(
            lambda x: "splice region" if x in ["splice donor", "splice acceptor"] else x
        )
        df["variant_type"] = df["variant_type"].apply(
            lambda x: "frameshift" if x in ["frameshift insertion", "frameshift deletion"] else x
        )

    if "variant_inheritance" in content_columns:
        # Recode "maternally inherited", "paternally inherited",
        # "maternally and paternally inherited homozygous" to "inherited"
        orig = ["maternally inherited", "paternally inherited", "maternally and paternally inherited homozygous"]
        df["variant_inheritance"] = df["variant_inheritance"].apply(lambda x: "inherited" if x in orig else x)

    return df


def restrict_truth_genes_to_output(truth_df: pd.DataFrame, output_genes: Set[str]) -> pd.DataFrame:
    print("Warning: restricting truth set to genes in the output set.")
    return truth_df[truth_df.gene.isin(output_genes)]


def restrict_output_papers_to_truth(output_df: pd.DataFrame, truth_papers: Set[str]) -> pd.DataFrame:
    print("Warning: restricting output set to papers in the truth set.")
    return output_df[output_df.pmid.isin(truth_papers)]


def sanity_check_inputs(
    truth_df: pd.DataFrame, output_df: pd.DataFrame, content_columns: Set[str], index_columns: Set[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    missing_from_truth = content_columns.union(index_columns) - set(truth_df.columns)
    if missing_from_truth:
        raise ValueError(f"Truth table is missing columns: {missing_from_truth}")

    missing_from_output = content_columns.union(index_columns) - set(output_df.columns)
    if missing_from_output:
        raise ValueError(f"Output table is missing columns: {missing_from_output}")

    idx_col_list = list(index_columns)
    # Ensure that the index columns are unique.
    if not truth_df.set_index(idx_col_list).index.is_unique:
        # Get a list of the non-unique indices
        non_unique_indices = truth_df[truth_df.duplicated(subset=idx_col_list, keep=False)][idx_col_list]
        # Print a warning and deduplicate.
        print(f"Warning: Truth table has non-unique index columns: {non_unique_indices}")
        print("Deduplicating truth table.")
        truth_df = truth_df[~truth_df.duplicated(subset=idx_col_list, keep="first")]

    if not output_df.set_index(idx_col_list).index.is_unique:
        # Get a list of the non-unique indices
        non_unique_indices = output_df[output_df.duplicated(subset=idx_col_list, keep=False)][idx_col_list]
        # Print a warning and deduplicate.
        print(f"Warning: Output table has non-unique index columns: {non_unique_indices}")
        print("Deduplicating output table.")
        output_df = output_df[~output_df.duplicated(subset=idx_col_list, keep="first")]

    return truth_df, output_df


@cache
def _get_variant_factory() -> HGVSVariantFactory:
    return DiContainer().create_instance({"di_factory": "lib/config/objects/variant_factory.yaml"}, {})


def _convert_single_to_three(single_code: str) -> str:
    """Convert a single letter amino acid code to three letter."""
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
    variant_factory = _get_variant_factory()

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


def normalize_hgvs_representations(df: pd.DataFrame, cols: List[str] | None = None) -> pd.DataFrame:
    """Normalize the HGVS representations in a dataframe.

    This ensures that the HGVS representations are in a consistent format. This includes using 3-letter amino acids
    for p. representations, and then normalizing the representations using the HGVSVariantFactory. This will allow
    for direct string comparison of the representations to check equivalence.
    """
    if cols is None:
        cols = ["hgvs_c", "hgvs_p"]

    for col in cols:
        if col == "hgvs_p":
            df["hgvs_p_orig"] = df["hgvs_p"]
            df["hgvs_p"] = df.apply(_normalize_hgvs_p, axis=1)
        elif col == "hgvs_c":
            df["hgvs_c_orig"] = df["hgvs_c"]
            df["hgvs_c"] = df.apply(_normalize_hgvs_c, axis=1)
    return df


def add_single_variant_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a single variant column to the dataframe."""
    df["hgvs_desc"] = df["hgvs_c"].fillna(df["hgvs_p"])
    return df


def consolidate_near_miss_individual_ids(
    mgt_df: pd.DataFrame, pipeline_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    for group, mgt_group_df in mgt_df.groupby(["paper_id", "hgvs_desc"]):
        # make a dict keyed on individual_id, with values that are the consolidated individual_ids.
        individual_id_map: dict[str, List[str]] = defaultdict(list)

        pipeline_group_df = pipeline_df[(pipeline_df["paper_id"] == group[0]) & (pipeline_df["hgvs_desc"] == group[1])]

        for individual_id in set(mgt_group_df.individual_id.unique()).union(pipeline_group_df.individual_id.unique()):
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

            mgt_df.loc[(mgt_df["paper_id"] == group[0]) & (mgt_df["hgvs_desc"] == group[1]), "individual_id"] = (
                mgt_group_df.individual_id.map(mapping)
            )
            pipeline_df.loc[
                (pipeline_df["paper_id"] == group[0]) & (pipeline_df["hgvs_desc"] == group[1]), "individual_id"
            ] = pipeline_df.individual_id.map(mapping)

    return mgt_df, pipeline_df


def merge_dfs(
    mgt_df: pd.DataFrame, pipeline_df: pd.DataFrame, columns_of_interest: Set[str], unique_columns: Set[str]
) -> pd.DataFrame:
    """Merge the two dataframes, keeping only the columns of interest.

    This function enforces that unique_columns is a unique MultiIndex for the two dataframes, then merges the two.
    Input dataframes remain unchanged.
    """
    mgt_df = mgt_df.copy().reset_index()
    mgt_df = mgt_df[[c for c in columns_of_interest if c in mgt_df.columns]]

    pipeline_df = pipeline_df.copy().reset_index()
    pipeline_df = pipeline_df[[c for c in columns_of_interest if c in pipeline_df.columns]]

    # Add a column for provenance.
    mgt_df["in_truth"] = True
    mgt_df["in_truth"] = mgt_df["in_truth"].astype("boolean")  # support nullable.
    pipeline_df["in_pipeline"] = True
    pipeline_df["in_pipeline"] = pipeline_df["in_pipeline"].astype("boolean")  # support nullable.

    # Merge the two dataframes.
    merged_df = pd.merge(
        mgt_df.drop_duplicates(subset=unique_columns).set_index(list(unique_columns)),
        pipeline_df.drop_duplicates(subset=unique_columns).set_index(list(unique_columns)),
        how="outer",
        left_index=True,
        right_index=True,
        suffixes=["_truth", "_output"],
    )

    merged_df["in_truth"] = merged_df["in_truth"].fillna(False)
    merged_df["in_pipeline"] = merged_df["in_pipeline"].fillna(False)

    if "gene_truth" in merged_df.columns:
        merged_df["gene"] = merged_df["gene_truth"].fillna(merged_df["gene_output"])
        merged_df.drop(columns=["gene_truth", "gene_output"], inplace=True)

    # Reorder columns, keeping in_truth and in_pipeline as the last two.
    merged_df = merged_df[
        [c for c in merged_df.columns if c not in {"in_truth", "in_pipeline"}] + ["in_truth", "in_pipeline"]
    ]

    return merged_df


def write_observation_finding_summary(merged_df: pd.DataFrame, output_filepath: str) -> None:
    precision = merged_df.in_truth[merged_df.in_pipeline].mean()
    recall = merged_df.in_pipeline[merged_df.in_truth].mean()

    print(f"Writing observation finding summary to: {output_filepath}")

    # Print the results to an output text file.
    with open(output_filepath, "w") as f:
        f.write("-- Observation finding summary performance --\n")
        f.write(f"  Observation finding N: {merged_df.shape[0]}\n")
        f.write(f"  Observation finding precision: {precision:.2f}\n")
        f.write(f"  Observation finding recall: {recall:.2f}\n")

        f.write("\n")
        f.write("-- Observation finding discrepancies --\n")

        if precision < 1 or recall < 1:
            printable_df = merged_df.reset_index()  #

            result = printable_df[~printable_df.in_truth | ~printable_df.in_pipeline].sort_values(
                ["gene", "pmid", "hgvs_desc"]
            )[
                [
                    c
                    for c in printable_df.columns
                    if c
                    in [
                        "hgvs_desc",
                        "gene",
                        "pmid",
                        "individual_id",
                        "hgvs_c_truth",
                        "hgvs_p_truth",
                        "hgvs_c_output",
                        "hgvs_p_output",
                        "in_truth",
                        "in_pipeline",
                    ]
                ]
            ]

            # TODO, better way to print this out?
            f.write(result.to_string())

            # result now available to view interactively.
        else:
            f.write("All observations found. This is likely because the Truthset observation finder was used.")


def write_observation_finding_outputs(merged_df: pd.DataFrame, output_dir: str) -> None:
    merged_df[["in_truth", "in_pipeline"]].to_csv(os.path.join(output_dir, "observation_finding_results.tsv"), sep="\t")


def _hpo_str_to_set(hpo_compound_string: str) -> Set[str]:
    """Convert a comma-separated list of HPO terms to a set of terms.

    Takes a string of the form "Foo (HP:1234), Bar (HP:4321) and provides a set of strings that correspond to the
    HPO IDs embedded in the string.
    """
    return set(re.findall(r"HP:\d+", hpo_compound_string)) if pd.isna(hpo_compound_string) is False else set()


@cache
def _get_ontology() -> Type[Ontology]:
    Ontology()
    return Ontology


def _generalize_hpo_term(hpo_term: str, depth: int = 3) -> str:
    """Take an HPO term ID and return the generalized version of that term at `depth`.

    `depth` determines the degree to which hpo_term gets generalized, setting depth=1 will always return HP:0000001.

    If the provided term is more generalized than depth (e.g., "HP:0000118"), then that term itself will be returned.
    If the provided term doesn't exist in the ontology, then an error will be raised.
    """
    try:
        hpo_obj = _get_ontology().get_hpo_object(hpo_term)
    except RuntimeError:
        # HPO term not found in pyhpo, can't use
        print("Warning: HPO term not found in pyhpo, can't use", hpo_term)
        return ""

    try:
        path_len, path, _, _ = _get_ontology().get_hpo_object("HP:0000001").path_to_other(hpo_obj)
    except RuntimeError:
        # No root found, occurs for obsolete terms.
        return hpo_obj.__str__()
    if path_len < depth:
        return hpo_obj.__str__()
    return path[depth - 1].__str__()


def _match_hpo_sets(
    hpo_left: str, hpo_right: str
) -> Tuple[list[str], Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    left_terms = _hpo_str_to_set(hpo_left)
    right_terms = _hpo_str_to_set(hpo_right)

    # Build a mapping from specific to general terms.
    left_gen_dict = {t: _generalize_hpo_term(t) for t in left_terms}
    right_gen_dict = {t: _generalize_hpo_term(t) for t in right_terms}

    # Invert the mapping.
    left_spec_dict: dict[str, list[str]] = defaultdict(list)
    for k, v in left_gen_dict.items():
        if k != "":
            left_spec_dict[v].append(k)

    right_spec_dict: dict[str, list[str]] = defaultdict(list)
    for k, v in right_gen_dict.items():
        if k != "":
            right_spec_dict[v].append(k)

    # Get the generalized terms for the left and right sets.
    left_gen = set(left_spec_dict.keys())
    right_gen = set(right_spec_dict.keys())

    matches = list(left_gen.intersection(right_gen))

    # Return the list of general terms that matched, a dict mapping each of those terms back to one or more specific
    # terms, a dict of terms that were in the left set but not the right, and a dict of terms that were in the right
    # set but not the left.

    match_dict = {k: list(set(left_spec_dict[k] + right_spec_dict[k])) for k in matches}
    left_dict = {k: left_spec_dict[k] for k in left_gen - right_gen}
    right_dict = {k: right_spec_dict[k] for k in right_gen - left_gen}
    return matches, match_dict, left_dict, right_dict


def evaluate_content_extraction(df: pd.DataFrame, columns_of_interest: Set[str]) -> pd.DataFrame:
    """Evaluate the content extraction performance.

    Modify the input datarame with a result column corresponding to each of the columns of
    interest. For most columns, this can just be a boolean indicator of whether the truth and output columns match.
    For the phenotype column, we'll need to do a fuzzy match based on HPO terms and we'll save the results from that
    matching operation.
    """
    for column in columns_of_interest:
        new_column = f"{column}_result"

        if column == "phenotype":
            # Phenotype matching can't be a direct string compare. We'll fuzzy match the HPO terms, note that we're
            # ignoring anything in here that couldn't be matched via HPO.
            df[new_column] = df.apply(
                lambda row: _match_hpo_sets(row["phenotype_truth"], row["phenotype_output"]), axis=1
            )
        else:
            # Other content columns are just string compares.
            df[new_column] = df[f"{column}_truth"].str.lower() == df[f"{column}_output"].str.lower()

    return df


INDICES_FOR_COLUMN = {
    "animal_model": ["gene", "pmid", "hgvs_desc", "individual_id"],
    "engineered_cells": ["gene", "pmid", "hgvs_desc", "individual_id"],
    "patient_cells_tissues": ["gene", "pmid", "hgvs_desc", "individual_id"],
    "phenotype": ["gene", "pmid", "hgvs_desc", "individual_id"],
    "study_type": ["gene", "pmid"],
    "variant_inheritance": ["gene", "pmid", "hgvs_desc", "individual_id"],
    "variant_type": ["gene", "pmid", "hgvs_desc"],
    "zygosity": ["gene", "pmid", "hgvs_desc", "individual_id"],
}


def get_eval_df(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df.empty:
        return df

    indices = INDICES_FOR_COLUMN[column]
    eval_df = df[~df.reset_index().set_index(indices).index.duplicated(keep="first")]
    return eval_df[[f"{column}_result", f"{column}_truth", f"{column}_output"]]


def write_content_extraction_summary(df: pd.DataFrame, columns_of_interest: Set[str], output_filepath: str) -> None:
    print(f"Writing content extraction summary to: {output_filepath}")

    # Print the results to an output text file.
    with open(output_filepath, "w") as f:
        for column in columns_of_interest:
            f.write(f"-- Content extraction summary for {column} --\n")

            eval_df = get_eval_df(df, column)

            if column != "phenotype":
                match = eval_df[f"{column}_result"]
                f.write(f"Content extraction accuracy for {column}: {match.mean():.3f} (of N={match.count()})\n")
                f.write("  Truth vs Pipeline output:\n")

                for idx, row in eval_df.iterrows():
                    if not match[idx]:  # type: ignore
                        f.write(f"!! Match ({idx}): {row[f'{column}_truth']} != {row[f'{column}_output']}\n")

            else:
                pheno_stats = eval_df["phenotype_result"]
                match = pheno_stats.apply(lambda x: len(x[2]) == 0 and len(x[3]) == 0)
                f.write(f"Content extraction accuracy for {column}: {match.mean():.3f} (of N={match.count()})\n")

                for idx, row in eval_df.iterrows():
                    if not match[idx]:  # type: ignore
                        f.write(f"##Mismatch ({idx})\n")
                        for i, x in enumerate(pheno_stats[idx]):  # type: ignore
                            if i == 1:
                                f.write(f"  Common: {x}\n")
                            elif i == 2:
                                f.write(f"  Truth only: {x}\n")
                            elif i == 3:
                                f.write(f"  Output only: {x}\n")
                        f.write(f"  Truth: {row[f'{column}_truth']}\n")
                        f.write(f"  Output: {row[f'{column}_output']}\n\n")
            f.write("\n")


def write_content_extraction_outputs(df: pd.DataFrame, output_dir: str) -> None:
    columns = sorted(df.columns)
    df[columns].to_csv(os.path.join(output_dir, "content_extraction_results.tsv"), sep="\t")


def _plot_confusion_matrix(
    truth: pd.Series, pipeline: pd.Series, labels: List[str], column: str, output_filepath: str
) -> None:
    """Plots a confusion matrix heatmap for two categorical series."""
    truth[~truth.isin(labels)] = "other"
    pipeline[~pipeline.isin(labels)] = "other"

    if any(truth == "other") or any(pipeline == "other"):
        labels.append("other")

    cm = confusion_matrix(truth, pipeline, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Output")
    plt.ylabel("Truth")
    plt.title(f"Confusion matrix for {column}")
    plt.show()
    plt.savefig(output_filepath)


PLOT_CONFIG = {
    "animal_model": {
        "options": ["true", "false"],
    },
    "engineered_cells": {
        "options": ["true", "false"],
    },
    "patient_cells_tissues": {
        "options": ["true", "false"],
    },
    "variant_type": {
        "options": [
            "missense",
            "stop gained",
            "splice region",
            "frameshift",
            "synonymous",
            "inframe deletion",
            "indel",
            "unknown",
            "failed",
        ],
    },
    "variant_inheritance": {
        "options": [
            "unknown",
            "de novo",
            "inherited",
            "failed",
        ],
    },
    "zygosity": {
        "options": ["none", "homozygous", "heterozygous", "compound heterozygous", "failed"],
    },
}


def write_content_plots(df: pd.DataFrame, columns_of_interest: Set[str], output_dir: str) -> None:

    print(f"Writing content extraction plots to: {output_dir}")

    for column in columns_of_interest:
        if column in PLOT_CONFIG:
            eval_df = get_eval_df(df, column)

            _plot_confusion_matrix(
                eval_df[f"{column}_truth"].copy(),
                eval_df[f"{column}_output"].copy(),
                PLOT_CONFIG[column]["options"].copy(),
                column,
                os.path.join(output_dir, f"{column}_confusion_matrix.png"),
            )


def main(args: argparse.Namespace) -> None:
    content_columns = set(args.content_columns.split(","))
    index_columns = {"gene", "individual_id", "hgvs_c", "hgvs_p", "pmid"}
    extra_columns = {"in_supplement"}

    # Gather inputs.
    ground_truth = get_mgt(args)
    pipeline_path, pipeline_outputs = get_pipeline_output(args)

    # Normalize individual IDs.
    ground_truth = normalize_individual_ids_in_df(ground_truth)
    pipeline_outputs = normalize_individual_ids_in_df(pipeline_outputs)

    # Recode field columns.
    ground_truth = recode_field_columns_in_df(ground_truth, content_columns)
    pipeline_outputs = recode_field_columns_in_df(pipeline_outputs, content_columns)

    # Subset truth or output sets if necessary.
    if args.restrict_truth_genes_to_output:
        ground_truth = restrict_truth_genes_to_output(ground_truth, set(pipeline_outputs.gene.unique()))
    if not args.include_non_truth_papers_in_output:
        pipeline_outputs = restrict_output_papers_to_truth(pipeline_outputs, set(ground_truth.pmid.unique()))

    # Sanity check inputs.
    ground_truth, pipeline_outputs = sanity_check_inputs(ground_truth, pipeline_outputs, content_columns, index_columns)

    # Normalize HGVS representations.
    # No need to normalize hgvs_c in the output df, since it's already normalized by the pipeline.
    ground_truth = normalize_hgvs_representations(ground_truth)
    pipeline_outputs = normalize_hgvs_representations(pipeline_outputs, ["hgvs_p"])

    # Add a single variant column, reconfigure our notion of column groups.
    ground_truth = add_single_variant_column(ground_truth)
    pipeline_outputs = add_single_variant_column(pipeline_outputs)
    index_columns -= {"hgvs_c", "hgvs_p"}
    index_columns.add("hgvs_desc")
    extra_columns.update({"hgvs_c", "hgvs_p"})

    # Consolidate near misses in the individual ID column on a per paper/variant basis.
    ground_truth, pipeline_outputs = consolidate_near_miss_individual_ids(ground_truth, pipeline_outputs)

    # Prep to write outputs.
    outdir = args.outdir or pipeline_path + "_content_extraction_benchmarks"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    all_columns = content_columns.union(index_columns).union(extra_columns)

    # Merge the dataframes treating observations as (variant, individual) tuples.
    merged_data = merge_dfs(ground_truth, pipeline_outputs, all_columns, index_columns).query("in_supplement != 'Y'")

    # Merge the dataframes considering variants only (i.e., discarding multiple individuals with the same variant).
    new_idx = index_columns - {"individual_id"}
    merged_var_data = merge_dfs(ground_truth, pipeline_outputs, all_columns, new_idx).query("in_supplement != 'Y'")

    # Write observation finding-related outputs.
    write_observation_finding_summary(merged_data, os.path.join(outdir, "observation_finding_summary.txt"))
    write_observation_finding_summary(merged_var_data, os.path.join(outdir, "variant_finding_summary.txt"))
    write_observation_finding_outputs(merged_data, outdir)

    # Assess content extraction performance.
    shared_df = merged_data[merged_data.in_truth & merged_data.in_pipeline].copy()  # Don't work with a view.
    content_result = evaluate_content_extraction(df=shared_df, columns_of_interest=content_columns)

    # Write content extraction-related outputs.
    write_content_extraction_summary(
        df=content_result,
        columns_of_interest=content_columns,
        output_filepath=os.path.join(outdir, "content_extraction_summary.txt"),
    )
    write_content_extraction_outputs(df=content_result, output_dir=outdir)

    # Write content extraction plots.
    write_content_plots(content_result, content_columns, outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evidence Aggregator Content Extraction Benchmarks")
    parser.add_argument(
        "-p",
        "--pipeline-output",
        nargs="?",
        default="",
        type=str,
        help="Path to the output directory corresponding to the pipeline run. "
        + "If none is provided, the most recent output for 'run_evagg_pipeline' will be used.",
    )
    parser.add_argument(
        "-m",
        "--mgt-train-test-path",
        nargs="?",
        default="data/v1/evidence_test_v1.tsv",
        type=str,
        help="Default is data/v1/evidence_test_v1.tsv.",
    )
    parser.add_argument(
        "-c",
        "--content-columns",
        nargs="?",
        default=(
            "phenotype,zygosity,variant_inheritance,variant_type,study_type,"
            "engineered_cells,patient_cells_tissues,animal_model"
        ),
        type=str,
        help=(
            "Comma-separated list of content columns to compare. Default is 'phenotype,zygosity,variant_inheritance,"
            "variant_type,study_type,engineered_cells,patient_cells_tissues,animal_model'."
        ),
    )
    parser.add_argument(
        "--restrict-truth-genes-to-output",
        action="store_true",
        help=(
            "Flag to restrict the truth set to genes in the output set. Set this to true for runs on a subset of genes "
            "from the truth set. Default is False."
        ),
    )
    parser.add_argument(
        "--include-non-truth-papers-in-output",
        action="store_true",
        help=(
            "Flag to include observations from papers that aren't in the truth set. Set this to True for pipeline "
            "runs that don't include paper finding. Default is False."
        ),
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default="",
        type=str,
        help=(
            "Results output directory. Default defaults to the pipeline input directory with _paper_finding appended."
        ),
    )
    args = parser.parse_args()

    print("Evidence Aggregator Content Extraction Benchmarks:")
    for arg, value in vars(args).items():
        print(f"- {arg}: {value}")

    print("\n")

    main(args)
