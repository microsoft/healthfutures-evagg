"""This script compares two evagg output tables, specifically focusing on the content extraction performance.

Content extraction has two logical components:
1. Identifying the observations (Tuple[variant, individual]) in the paper.
2. Extracting the content associated with those observations.

This notebook compares the performance of the two components separately.
"""

# %% Imports.

import os
from typing import Any, Set

import pandas as pd

from lib.evagg.content import HGVSVariantFactory
from lib.evagg.ref import MutalyzerClient, NcbiReferenceLookupClient
from lib.evagg.svc import CosmosCachingWebClient, get_dotenv_settings

# %% Constants.

TRUTH_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "v1", "evidence_train_v1.tsv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", ".out", "observation_benchmark.tsv")

# TODO: after we rethink variant nomenclature, figure out whether we need to check the hgvs nomenclatures for agreement.
CONTENT_COLUMNS: Set[str] = set()  # when CONTENT_COLUMNS is empty we're just comparing observation-finding
# CONTENT_COLUMNS = {"phenotype", "zygosity", "variant_inheritance"} # noqa
INDEX_COLUMNS = {"individual_id", "hgvs_c", "hgvs_p", "paper_id"}
EXTRA_COLUMNS = {"gene", "in_supplement"}

# TODO, just get the gene list from the yaml?
RESTRICT_TRUTH_GENES_TO_OUTPUT = False  # if True, only compare the genes in the output set to the truth set.

# %% Read in the truth and output tables.

truth_df = pd.read_csv(TRUTH_PATH, sep="\t")
if "doi" in truth_df.columns:
    print("Warning: converting doi to paper_id")
    truth_df.rename(columns={"doi": "paper_id"}, inplace=True)

output_df = pd.read_csv(OUTPUT_PATH, sep="\t", skiprows=1)

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
refseq_client = NcbiReferenceLookupClient(web_client)
variant_factory = HGVSVariantFactory(
    validator=mutalyzer_client, normalizer=mutalyzer_client, refseq_client=refseq_client
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

    if (variant_obj is None or not variant_obj.valid) and hgvs_desc.startswith("p."):
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


if "hgvs_c" in truth_df.columns:
    truth_df["hgvs_c_orig"] = truth_df["hgvs_c"]
    truth_df["hgvs_c"] = truth_df.apply(_normalize_hgvs_c, axis=1)

if "hgvs_p" in truth_df.columns:
    truth_df["hgvs_p_orig"] = truth_df["hgvs_p"]
    truth_df["hgvs_p"] = truth_df.apply(_normalize_hgvs_p, axis=1)

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

    # Reset all_columns
    all_columns = CONTENT_COLUMNS.union(INDEX_COLUMNS).union(EXTRA_COLUMNS)


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

# %% Assess variant finding.

precision = merged_df.in_truth[merged_df.in_output == True].mean()
recall = merged_df.in_output[merged_df.in_truth == True].mean()

# Make a copy of merged_df removing all rows where in_supplement is 'Y'
merged_df_no_supplement = merged_df[merged_df.in_supplement != "Y"]
precision_no_supplement = merged_df_no_supplement.in_truth[merged_df_no_supplement.in_output == True].mean()
recall_no_supplement = merged_df_no_supplement.in_output[merged_df_no_supplement.in_truth == True].mean()

print("---- Variant finding performance ----")
print("Overall")
print(f"  Variant finding precision: {precision:.2f}")
print(f"  Variant finding recall: {recall:.2f}")
print("Ignoring truth papers from supplement")
print(f"  Variant finding precision: {precision_no_supplement:.2f}")
print(f"  Variant finding recall: {recall_no_supplement:.2f}")
print()

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

merged_df[merged_df.in_supplement != "Y"].sort_values("gene")

# %% Assess content extraction.

if CONTENT_COLUMNS:
    shared_df = merged_df[merged_df.in_truth & merged_df.in_output]

    print("---- Content extraction performance ----")

    for column in CONTENT_COLUMNS:
        match = shared_df[f"{column}_truth"].str.lower() == shared_df[f"{column}_output"].str.lower()
        print(f"Content extraction accuracy for {column}: {match.mean()}")

        for idx, row in shared_df[~match].iterrows():
            print(f"  Mismatch ({idx}): {row[f'{column}_truth']} != {row[f'{column}_output']}")
        print()


# %%
