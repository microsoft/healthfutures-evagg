# %% Imports.
import time
import urllib.parse
from typing import Any, Dict

import requests

# %% Functions.


def hgvs_to_vcf(hgvs: str) -> Dict[str, str] | None:
    encoded = urllib.parse.quote(hgvs)
    # 2 queries per second max.
    url = f"https://rest.variantvalidator.org/VariantValidator/variantvalidator/GRCh38/{encoded}/select/?content-type=application/json"
    response = requests.get(url)
    data = response.json()

    assert len(data) == 3, f"Unexpected number of keys in VV response: {url}"

    variant_info = data[next(key for key in data.keys() if key not in ["metadata", "flag"])]

    return variant_info.get("primary_assembly_loci", {}).get("grch38", {}).get("vcf", {})
    # TODO handle status code 429


def join_popmax_faf95(vcf: Dict[str, str]) -> float | None:
    gnomad = gnomad_for_vcf(vcf)
    if gnomad is None:
        return None
    if "errors" in gnomad:
        if gnomad["errors"][0]["message"] == "Variant not found":
            return None
        else:
            raise Exception(f"Unexpected error response from gnomAD API: {gnomad}")
    return gnomad["data"]["joint"]["faf95"]["popmax"]


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
    flags
    lof_curations {
      gene_id
      gene_symbol
      verdict
      flags
      project
    }
    rsids
    transcript_consequences {
      domains
      gene_id
      gene_version
      gene_symbol
      hgvs
      hgvsc
      hgvsp
      is_canonical
      is_mane_select
      is_mane_select_version
      lof
      lof_flags
      lof_filter
      major_consequence
      polyphen_prediction
      sift_prediction
      transcript_id
      transcript_version
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
    response = requests.post(url, json={"query": query, "variables": variables})
    if response.status_code == 200:
        return response.json()  # Returns the JSON response
    else:
        raise Exception(f"Query failed to run by returning code of {response.status_code}. {response.text}")

    # TODO, handle
    # {'alt': 'T', 'chr': '5', 'pos': '128338938', 'ref': 'C'}
    # {'errors': [{'message': 'Variant not found'}], 'data': {'variant': None}} - status code 200


# %% Do stuff.
result1 = gnomad_for_vcf({"chr": "1", "pos": "55052746", "ref": "GT", "alt": "G"})

# time.sleep(6)
# result2 = gnomad_for_vcf({"chr": "1", "pos": "55058620", "ref": "TG", "alt": "T"})

# %%
