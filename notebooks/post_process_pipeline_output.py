"""Post process pipeline output in preparation for seqr load."""

# %% Imports.

import os
import time
import urllib.parse
from functools import cache
from typing import Any, Dict

import pandas as pd

from lib.di import DiContainer

# %% Constants.

PIPELINE_OUTPUT = ".out/run_evagg_pipeline_20240701_025327/pipeline_benchmark.tsv"
VCF_VARIANT_FORMAT_MAPPING_DIR = "notebooks/post_processing_vcf/"

# %% Load pipeline output.

df = pd.read_csv(PIPELINE_OUTPUT, sep="\t", header=1)
df.set_index("evidence_id", inplace=True)

# %% Handle nans in output

df.fillna({"hgvs_c": "", "hgvs_p": "", "validation_error": ""}, inplace=True)

# %% Handle validation errors.

droppable_errors = ["ESYNTAXUC", "EOUTOFBOUNDARY"]

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

# %% Determine gnomad allele frequencies for each variant.

web_client = DiContainer().create_instance(
    spec={"di_factory": "lib/config/objects/web_cache.yaml", "web_settings": {"max_retries": 0}}, resources={}
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


# @cache
# def hgvs_to_vcf(hgvs: str) -> Dict[str, str] | None:
#     encoded = urllib.parse.quote(hgvs)
#     # 2 queries per second max.
#     url = f"https://rest.variantvalidator.org/VariantValidator/variantvalidator/GRCh38/{encoded}/select/?content-type=application/json"
#     data = web_client.get(url, content_type="json")

#     assert len(data) == 3, f"Unexpected number of keys in VV response: {url}"

#     variant_info = data[next(key for key in data.keys() if key not in ["metadata", "flag"])]

#     return variant_info.get("primary_assembly_loci", {}).get("grch38", {}).get("vcf", {})
#     # TODO handle status code 429


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
    return web_client.post(url, data={"query": query, "variables": variables}, content_type="json")

    # TODO, handle
    # {'alt': 'T', 'chr': '5', 'pos': '128338938', 'ref': 'C'}
    # {'errors': [{'message': 'Variant not found'}], 'data': {'variant': None}} - status code 200


# %%
for _, row in df.iterrows():
    loop_start = time.time()
    if not row.hgvs_c or not row.transcript or row.validation_error:
        row.gnomad_frequency = ""
        continue

    print(f"Obtaining gnomAD allele frequency for: {row.transcript}:{row.hgvs_c}")
    # print(f"Converting HGVS to VCF: {row.transcript}:{row.hgvs_c}")
    vcf = hgvs_to_vcf(f"{row.transcript}:{row.hgvs_c}")
    # print(f"  {vcf['chr']}-{vcf['pos']}-{vcf['ref']}-{vcf['alt']}")
    if vcf is None:
        row.gnomad_frequency = ""
        continue
    start = time.time()
    # print(f"Extracting gnomAD allele frequency: {vcf}")
    freq = joint_popmax_faf95(vcf)
    if not freq:
        row.gnomad_frequency = "0"
    else:
        row.gnomad_frequency = f"{freq:.3g}"

    # print(f"  {freq}")
    wait_triggered = (call_elapsed := time.time() - start) > 0.05
    if wait_triggered:
        print(f"wait triggered: {vcf}")
        while time.time() - loop_start < 6:
            time.sleep(0.2)

# %%
