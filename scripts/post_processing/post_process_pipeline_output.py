"""Post process pipeline output in preparation for seqr load.

Execution of this post-processing script requires manually executing batch validation with the variantvalidator
website to obtain HGVS->VCF mappings. Running this script with an incomplete set of mappings will cause the required
mappings to be printed out and no allele frequencies will be obtained. If no required mappings are missing, then the
script will call the gnomAD web service to obtain AFs.

Because of this, this script will likely need to be run twice for any given output file. Once to obtain the list of 
missing HGVS->VCF mappings and another time to obtain the allele frequencies.
"""

# %% Imports.

import os
import time
from functools import cache
from typing import Any, Dict

import pandas as pd

from lib.di import DiContainer

# %% Constants.

PIPELINE_OUTPUT = ".out/run_evagg_pipeline_20240703_194329/pipeline_benchmark.tsv"
VCF_VARIANT_FORMAT_MAPPING_DIR = "scripts/post_processing/"

DROP_VALIDATION_ERRORS = True

# %% Helper functions

web_client = DiContainer().create_instance(
    spec={"di_factory": "lib/config/objects/web_cache.yaml", "web_settings": {"max_retries": 5}}, resources={}
)


@cache
def get_mapping_df() -> pd.DataFrame:
    vv_df = pd.read_csv(
        os.path.join(VCF_VARIANT_FORMAT_MAPPING_DIR, "vv_output_latest.tsv"), sep="\t", header=2
    ).reset_index()
    vv_df.drop_duplicates(subset=["Input"], inplace=True)
    vv_df.fillna(
        {"GRCh38_CHR": "", "GRCh38_POS": "", "GRCh38_REF": "", "GRCh38_ALT": "", "Gene_Symbol": ""}, inplace=True
    )
    return vv_df[["Input", "Warnings", "GRCh38_CHR", "GRCh38_POS", "GRCh38_REF", "GRCh38_ALT", "Gene_Symbol"]]


@cache
def hgvs_to_vcf(hgvs: str) -> Dict[str, str] | None:
    mapping_df = get_mapping_df()
    row = mapping_df[mapping_df["Input"] == hgvs]
    if row.empty:
        return None
    if row.shape[0] > 1:
        print(f"Warning: multiple rows for {hgvs}. Keeping first")
    row = row.iloc[0]
    if not row.GRCh38_CHR or not row.GRCh38_POS or not row.GRCh38_REF or not row.GRCh38_ALT:
        return None
    return {
        "chr": row["GRCh38_CHR"],
        "pos": str(int(row["GRCh38_POS"])),
        "ref": row["GRCh38_REF"],
        "alt": row["GRCh38_ALT"],
    }


def joint_popmax_faf95(vcf: Dict[str, str]) -> float | None:
    gnomad = gnomad_for_vcf(vcf)
    if gnomad is None:
        return None
    if "errors" in gnomad:
        if gnomad["errors"][0]["message"] == "Variant not found":
            return None
        else:
            raise Exception(f"Unexpected error response from gnomAD API: {gnomad}")
    return gnomad["data"]["variant"]["joint"]["faf95"]["popmax"]


def gnomad_for_vcf(vcf: Dict[str, str]) -> Dict[str, Any] | None:

    # GraphQL endpoint, 10 queries per minute max.
    url = "https://gnomad.broadinstitute.org/api"

    # The GraphQL query template
    query = """
query GnomadVariant($variantId: String!, $datasetId: DatasetId!) {
  variant(variantId: $variantId, dataset: $datasetId) {
    variant_id
    reference_genome
    chrom
    pos
    ref
    alt
    colocated_variants
    joint {
      faf95 {
        popmax
        popmax_population
      }
    }
  }
}
"""

    # Function to query for a single variant
    dataset_id = "gnomad_r4"
    variant_id = f"{vcf['chr']}-{vcf['pos']}-{vcf['ref']}-{vcf['alt']}"

    # Query variables
    variables = {"variantId": variant_id, "datasetId": dataset_id}

    # HTTP POST request
    return web_client.get(url, data={"query": query, "variables": variables}, content_type="json")


# %% Load pipeline output.

df = pd.read_csv(PIPELINE_OUTPUT, sep="\t", header=1)
df.set_index("evidence_id", inplace=True)

# %% Drop entirely duplicated rows.

# Check for duplicaes, if they exist issue a warning and drop.
if df.duplicated().any():
    print("WARNING: Duplicates found. Dropping duplicates.")
    df.drop_duplicates(inplace=True)

# %% Handle nans in output

df.fillna({"hgvs_c": "", "hgvs_p": "", "validation_error": ""}, inplace=True)

# %% Check to see whether all the hgvs_c values of interest are in the mapping file.

missing_variants = []
found_variants = []
vcf_mapping_df = get_mapping_df()

for _, row in df.iterrows():
    if not row.hgvs_c or not row.transcript or row.validation_error:
        continue

    var_str = f"{row.transcript}:{row.hgvs_c}"
    if not any(vcf_mapping_df.Input == var_str):
        missing_variants.append(var_str)
    else:
        found_variants.append(var_str)

if missing_variants:
    print(
        """
The following variants were missing from the vcf mapping file. Update this file using
the variant validator service and rerun this script.
"""
    )
    for mv in missing_variants:
        print(mv)

# %% Determine gnomad allele frequencies for each variant.

n_rows = df.shape[0]
counter = 0

if not missing_variants:
    for _, row in df.iterrows():
        counter += 1
        loop_start = time.time()
        if not row.hgvs_c or not row.transcript or row.validation_error:
            df.loc[row.name, "gnomad_frequency"] = ""  # type: ignore
            continue

        print(f"{counter} of {n_rows} - Obtaining gnomAD allele frequency for: {row.transcript}:{row.hgvs_c}")
        vcf = hgvs_to_vcf(f"{row.transcript}:{row.hgvs_c}")
        if vcf is None:
            df.loc[row.name, "gnomad_frequency"] = ""  # type: ignore
            continue
        start = time.time()
        freq = joint_popmax_faf95(vcf)
        if not freq:
            df.loc[row.name, "gnomad_frequency"] = "0"  # type: ignore
        else:
            df.loc[row.name, "gnomad_frequency"] = f"{freq:.3g}"  # type: ignore

        # Make a best guess as to whether the underlying API was called or the result was served from the cosmos cache.
        # Better to be conservative here (assuming we called the API when we didn't) than to be too optimistic.
        # See issues/93.
        wait_triggered = (call_elapsed := time.time() - start) > 0.05
        if wait_triggered:
            print(f"Waiting to recall gnomAD API: {vcf} [call_elapsed = {call_elapsed}]")
            while time.time() - loop_start < 6:
                time.sleep(0.2)
else:
    print("WARNING: not obtaining gnomad frequencies due to missing variants.")

# %% Handle validation errors.

if DROP_VALIDATION_ERRORS:
    droppable_errors = ["ESYNTAXUC", "EOUTOFBOUNDARY"]
    print("Dropping validation errors:", droppable_errors)
    df = df[~df["validation_error"].isin(droppable_errors)]

# Recode the validation error column with the following string replacements.
# "ESEQUENCEMISMATCH" -> "ESEQUENCEMISMATCH: Reference base mismatch."
# "EAMINOACIDMISMATCH" -> "EAMINOACIDMISMATCH: Reference amino acid mismatch."
# "EOUTOFBOUNDARY" -> "EOUTOFBOUNDARY: supplied position outside of sequence boundary."
# "ESYNTAXUC" -> "ESYNTAXUC: Syntax error in hgvs description."
# "ESYNTAXUEOF -> "ESYNTAXUEOF: Syntax error in hgvs description."
# "ERETR" -> "ERETR: Mismatch in supplied transcript."
#
# Additionally, replace any other validation errors with "Other validation error."
replacements = {
    "ESEQUENCEMISMATCH": "ESEQUENCEMISMATCH: Reference base mismatch.",
    "EAMINOACIDMISMATCH": "EAMINOACIDMISMATCH: Reference amino acid mismatch.",
    "EOUTOFBOUNDARY": "EOUTOFBOUNDARY: supplied position outside of sequence boundary.",
    "ESYNTAXUC": "ESYNTAXUC: Syntax error in hgvs description.",
    "ESYNTAXUEOF": "ESYNTAXUEOF: Syntax error in hgvs description.",
    "ERETR": "ERETR: Mismatch in supplied transcript.",
}

others = set(df["validation_error"].unique()) - set(replacements.keys())
others -= {""}

replacements.update({other: "Other validation error." for other in others})

df["validation_error"] = df["validation_error"].replace(replacements)

# %% Save the output.

post_processed_file = PIPELINE_OUTPUT.replace(".tsv", "_post_processed.tsv")
df.to_csv(post_processed_file, sep="\t")
