"""This notebook is for playing around with the hgvs package."""

# Ideas to explore:
#  - gene proximity to variants
#  - prompt-based detection of gene mentions within a paper

# %% Imports.

# import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Set

import requests

from lib.evagg.content import HGVSVariantFactory, HGVSVariantMention, VariantTopic
from lib.evagg.ref import MutalyzerClient, NcbiLookupClient, NcbiReferenceLookupClient
from lib.evagg.svc import CosmosCachingWebClient, RequestsWebContentClient, get_dotenv_settings
from lib.evagg.types import Paper

logger = logging.getLogger(__name__)


# %% Collect a bunch of variant names and corresponding genes from papers, it's ok of these are noisy, with
# incorrect gene associations (for example), since we're just going to use them as a smoke test
# for HGVS parsing.


# Helper utility function for getting gene symbols from gene IDs. Not currently supported by evagg. Also a candidate for
# incorporating into the PR, but not necessarily required in this case.
def gene_symbols_for_id(gene_ids: Sequence[str], max: int = -1) -> Dict[str, List[str]]:
    seq_str = ",".join(gene_ids)

    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gene&id={seq_str}&format=json"
    response = requests.get(url)
    response.raise_for_status()

    response_dict = response.json()
    symbols: Dict[str, List[str]] = {}

    for gene_id in gene_ids:
        gene = response_dict["result"].get(gene_id, None)
        if gene:
            symbol_list = [gene["name"]] + gene["otheraliases"].split(", ")
            if max >= 0:
                symbol_list = symbol_list[:max]
            symbols[gene_id] = symbol_list

    return symbols


PMCIDS = ["PMC6912785", "PMC6175136"]
kept_genes = ["COQ2", "DLD", "DGUOK"]

papers = [Paper(**{"id": pmcid, "is_pmc_oa": True, "pmcid": pmcid}) for pmcid in PMCIDS]

# We'll pubtator to get variants for these PMIDs.
lookup_client = NcbiLookupClient(RequestsWebContentClient(), settings=get_dotenv_settings(filter_prefix="NCBI_EUTILS_"))

refseq_finder = re.compile(r"N[MPGC]_[0-9]+\.[0-9]+")

# gene_client = NcbiGeneClient(BioEntrezClient(BioEntrezDotEnvConfig()))


def get_nearest_refseq(tx_matches: Sequence[Any], loc: int, max_dist: int) -> str | None:
    closest_str = None
    closest_distance = max_dist + 1
    for match in tx_matches:
        distance = min(abs(x - loc) for x in match.span())
        if distance < closest_distance:
            closest_str = match.group()
            closest_distance = distance
    return closest_str


variant_tuples = []

for idx, paper in enumerate(papers):
    anno = lookup_client.annotate(paper)
    if not anno:
        continue
    print(f"Analyzing paper {idx} - {paper}")

    # Preprocess the annotations to get a gene symbol lookup
    gene_ids = set()
    for p in anno["passages"]:
        for a in p["annotations"]:
            if a["infons"]["type"] == "Gene":
                # sometimes they're semicolon delimited
                gene_ids.update(a["infons"]["identifier"].split(";"))

    gene_symbol_dict = gene_symbols_for_id(list(gene_ids), max=1)

    for p in anno["passages"]:
        refseq_matches = [r for r in re.finditer(refseq_finder, p["text"])]  # noqa: C416
        for a in p["annotations"]:
            if a["infons"]["type"] == "Variant":
                vt = {"text": a["text"]}

                vt["hgvs"] = a["infons"].get("hgvs", None)

                gene_id_int = a["infons"].get("gene_id", None)
                vt["gene_id"] = gene_id_int

                if gene_id_int:
                    matching_symbols = gene_symbol_dict.get(str(gene_id_int), None)
                    if matching_symbols:
                        vt["gene"] = matching_symbols[0]
                    else:
                        vt["gene"] = None
                else:
                    vt["gene"] = None
                if vt["gene"] and vt["gene"] not in kept_genes:
                    continue

                if refseq_matches:
                    # TODO, this should actually filter based on the variant type (c., p., etc and only look for the
                    # correct refseq type)
                    vt["refseq"] = get_nearest_refseq(refseq_matches, a["locations"][0]["offset"] - p["offset"], 100)
                else:
                    vt["refseq"] = None

                variant_tuples.append(vt)

    break

# %% Try to assemble all of the topics based on the extracted topics above.

ref_seq_lookup_client = NcbiReferenceLookupClient()

#   web_client:
#     # di_factory: lib.evagg.svc.RequestsWebContentClient
#     di_factory: lib.evagg.svc.CosmosCachingWebClient
#     cache_settings:
#       di_factory: lib.evagg.svc.get_dotenv_settings
#       filter_prefix: "EVAGG_CONTENT_CACHE_"
#     web_settings:
#       max_retries: 3

web_client = CosmosCachingWebClient(
    get_dotenv_settings(filter_prefix="EVAGG_CONTENT_CACHE_"), web_settings={"max_retries": 0, "retry_codes": []}
)
mutalyzer_client = MutalyzerClient(web_client)

variant_factory = HGVSVariantFactory(
    normalizer=mutalyzer_client, back_translator=mutalyzer_client, refseq_client=ref_seq_lookup_client
)
variant_topics: List[VariantTopic] = []

for vt in variant_tuples:
    print(vt)
    try:
        v = variant_factory.try_parse(
            text_desc=vt["hgvs"] if vt["hgvs"] else vt["text"], gene_symbol=vt["gene"], refseq=vt["refseq"]
        )
    except ValueError as e:
        print(f"Error parsing variant {vt['text']}: {e}")
        continue
    vm = HGVSVariantMention(text=vt["text"], context="Some words", variant=v)
    # Check all the topics for a match, if none, add a new topic.
    found = False
    for topic in variant_topics:
        if topic.match(vm):
            topic.add_mention(vm)
            found = True
            break
    if not found:
        variant_topics.append(VariantTopic([vm], normalizer=mutalyzer_client, back_translator=mutalyzer_client))

# %%
