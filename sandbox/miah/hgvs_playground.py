"""This notebook is for playing around with the hgvs package."""

# Ideas to explore:
#  - gene proximity to variants
#  - prompt-based detection of gene mentions within a paper

# %% Imports.

import json
import logging
import os
import re
from dataclasses import dataclass
from functools import cache
from typing import Any, Dict, List, Sequence, Set

import requests

from lib.evagg.ref import NcbiLookupClient
from lib.evagg.svc import RequestsWebContentClient, get_dotenv_settings
from lib.evagg.types import Paper

logger = logging.getLogger(__name__)


# %% Define ReqSeqReference class.


class RefSeqReference:
    """Determine RefSeq 'Reference Standard' accessions for genes using the NCBI RefSeqGene database."""

    _ref: Dict[str, Dict[str, str]]
    _lazy_initialized: bool
    _DEFAULT_REFERENCE_DIR = "/home/azureuser/repos/ev-agg-exp/.ref/"  # TODO, change to be relative to repo root.
    _NCBI_REFERENCE_URL = "https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/RefSeqGene/LRG_RefSeqGene"

    def __init__(self, reference_dir: str = _DEFAULT_REFERENCE_DIR) -> None:
        # Lazy initialize so the constructor is fast.
        self._reference_dir = reference_dir
        self._lazy_initialized = False
        self._ref = {}
        pass

    def get_transcript_accession_for_symbol(self, symbol: str) -> str | None:
        """Get the RefSeq transcript accession for a gene symbol."""
        if not self._lazy_initialized:
            self._lazy_init()
        return self._ref.get(symbol, {}).get("RNA", None)

    def get_protein_accession_for_symbol(self, symbol: str) -> str | None:
        """Get the RefSeq protein accession for a gene symbol."""
        if not self._lazy_initialized:
            self._lazy_init()
        return self._ref.get(symbol, {}).get("Protein", None)

    def _download_reference(self, url: str, target: str) -> None:
        # Download the reference TSV file from NCBI.
        response = requests.get(url)
        response.raise_for_status()

        if response.status_code != 200:
            raise ValueError(f"Failed to download reference file from {url}")

        with open(target, "wb") as f:
            f.write(response.content)

    def _lazy_init(self) -> None:
        # Download the reference file if necessary.
        if not os.path.exists(self._reference_dir):
            logging.info(f"Creating reference directory at {self._reference_dir}")
            os.makedirs(self._reference_dir)

        reference_filepath = os.path.join(self._reference_dir, "LRG_RefSeqGene.tsv")
        if not os.path.exists(reference_filepath):
            logging.info(f"Downloading reference file to {reference_filepath}")
            self._download_reference(self._NCBI_REFERENCE_URL, reference_filepath)

        # Load the reference file into memory.
        logging.info(f"Loading reference file from {reference_filepath}")
        self._ref = self._load_reference(reference_filepath)
        self._lazy_initialized = True

    def _load_reference(self, filepath: str) -> Dict[str, Dict[str, str]]:
        """Load a reference TSV file into a dictionary."""
        with open(filepath, "r") as f:
            lines = f.readlines()

        header = lines[0].strip().split("\t")

        # First two columns are taxon and gene ID, which we don't care about, so we'll skip them.
        # The third column is gene symbol, which we'll use as a key.
        # Only keep rows where the last column is "reference standard". If there's more than
        # one row per gene symbol, print a warning and keep the first one.
        kept_fields = ["GeneID", "Symbol", "RSG", "RNA", "Protein"]
        reference_dict = {}
        for line in lines[1:]:
            fields = line.strip().split("\t")
            if fields[-1] != "reference standard":
                continue
            gene_symbol = fields[2]
            if gene_symbol in reference_dict:
                logging.info(f"Multiple reference standard entries for gene {gene_symbol}. Keeping the first one.")
                continue
            reference_dict[gene_symbol] = {k: v for k, v in zip(header, fields) if k in kept_fields}

        return reference_dict


# %% Define the Mutalyzer Reference class


class MutalyzerReference:
    def __init__(self) -> None:
        pass

    @cache
    def back_translate(self, hgvsp: str) -> Sequence[str]:
        """Back translate a protein variant description to a coding variant description using Mutalyzer.

        hgvsp: The protein variant description to back translate. Must conform to HGVS nomenclature.
        """
        url = f"https://mutalyzer.nl/api/back_translate/{hgvsp}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    @cache
    def normalize(self, hgvs: str) -> Dict[str, Any]:
        """Normalize an HGVS description using Mutalyzer.

        hgvs: The HGVS description to normalize, e.g., NM_000551.3:c.1582G>A
        """
        url = f"https://mutalyzer.nl/api/normalize/{hgvs}"
        response = requests.get(url)

        # Response code of 422 signifies an unprocessable entity.
        # This occurs when the description is syntactically invalid, but also
        # occurs when the description is biologically invalid (e.g., the reference is incorrect).
        # Detailed information is available in the response body, but it's not currently relevant.
        if response.status_code == 200:
            return response.json()
        else:
            return {}


# %% Build some useful tools

ref_seq_reference = RefSeqReference()

mutalyzer_reference = MutalyzerReference()


# %% Define the Variant, VariantMention, and VariantMentionGroup classes.


# Variant is a dataclass
@dataclass(frozen=True)
class HGVSVariant:
    """A representation of a genetic variant."""

    hgvs_desc: str
    gene_symbol: str | None
    refseq: str
    refseq_predicted: bool
    valid: bool

    def __str__(self) -> str:
        """Obtain a string representation of the variant."""
        return f"{self.refseq}:{self.hgvs_desc}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HGVSVariant):
            return False
        return (
            self.refseq == other.refseq
            and self.gene_symbol == other.gene_symbol
            and self.hgvs_no_paren() == other.hgvs_no_paren()
        )

    def hgvs_no_paren(self) -> str:
        """Return a string representation of the variant description without prediction parentheses.

        For example: p.(Arg123Cys) -> p.Arg123Cys
        """
        return self.hgvs_desc.replace("(", "").replace(")", "")


class HGVSVariantFactory:
    _mutalyzer: MutalyzerReference  # TODO, DI
    MITO_REFSEQ = "NC_012920.1"

    def __init__(self) -> None:
        # TODO, DI
        self._mutalyzer = MutalyzerReference()

    def _predict_refseq(self, text_desc: str, gene_symbol: str | None) -> str | None:
        """Predict the RefSeq for a variant based on its description and gene symbol."""
        if text_desc.startswith("p.") and gene_symbol:
            return ref_seq_reference.get_protein_accession_for_symbol(gene_symbol)
        elif text_desc.startswith("c.") and gene_symbol:
            return ref_seq_reference.get_transcript_accession_for_symbol(gene_symbol)
        elif text_desc.startswith("m."):
            return self.MITO_REFSEQ
        elif text_desc.startswith("g."):
            raise ValueError(f"Genomic (g. prefixed) variants must have a RefSeq. None was provided for {text_desc}")
        else:
            raise ValueError(f"Unsupported HGVS type: {text_desc} with gene symbol {gene_symbol}")

    def try_parse(self, text_desc: str, gene_symbol: str | None, refseq: str | None) -> HGVSVariant:
        """Attempt to parse a variant based on description and an optional gene symbol and optional refseq.

        `gene_symbol` is required for protein (p.) and coding (c.) variants, but not mitochondrial (m.) or genomic (g.)
        variants.

        If not provided, `refseq` will be predicted based on the variant description and gene symbol for protein and
        coding variants. `refseq` is required for genomic (g.) variants.

        Raises a ValueError if the above requirements are not met, or if anything but (g., c., p., m.) is provided as
        the description prefix.

        Raises a ValueError if the variant description is not syntactically valid according to HGVS nomenclature.

        Raises a ValueError if the refseq and variant description are not compatible (e.g., a protein variant on a
        transcript refseq).
        """
        # If no refseq is provided, we'll try to predict it.
        if not refseq:
            refseq = self._predict_refseq(text_desc, gene_symbol)
            refseq_predicted = True
        else:
            refseq_predicted = False

        if not refseq:
            raise ValueError(
                f"RefSeq required for all variants. None was provided or predicted for {text_desc, gene_symbol}"
            )

        # Validate the variant for both syntactic and biological correctness.
        is_valid = self._validate_variant(refseq, text_desc)

        return HGVSVariant(
            hgvs_desc=text_desc,
            gene_symbol=gene_symbol,
            refseq=refseq,
            refseq_predicted=refseq_predicted,
            valid=is_valid,
        )

    def _validate_variant(self, refseq: str, text_desc: str) -> bool:
        # Validate the variant. Returns True if the variant is valid, False otherwise.
        return bool(self._mutalyzer.normalize(f"{refseq}:{text_desc}"))


@dataclass(frozen=True)
class HGVSVariantMention:
    """A representation of a text mention of a genetic variant in a paper."""

    text: str
    context: str  # TODO: Maybe should be a reference to a Passage object?
    variant: HGVSVariant


class VariantTopic:
    """A Representation of a topic in a paper pertaining to a genetic variant."""

    _variant_mentions: List[HGVSVariantMention]
    _variants: Set[HGVSVariant]

    def __init__(self, mentions: Sequence[HGVSVariantMention]) -> None:
        self._variant_mentions = []
        self._variants = set()

        for mention in mentions:
            self.add_mention(mention)

    def __str__(self) -> str:
        return f"VariantTopic: {', '.join(str(v) for v in self._variants)}"

    def get_mentions(self) -> Sequence[HGVSVariantMention]:
        """Get all variant mentions in the topic."""
        return self._variant_mentions

    def add_mention(self, mention: HGVSVariantMention) -> None:
        """Add a mention to the topic."""
        # Add the mention to the list of mentions.
        self._variant_mentions.append(mention)

        # Additionally, add the variant to the set of variants if it's not already there.
        self._variants.add(mention.variant)

    def get_variants(self) -> List[HGVSVariant]:
        """Get all variants mentioned in the topic."""
        return list(self._variants)

    def _linked(self, a: HGVSVariant, b: HGVSVariant) -> bool:
        """Determine if two variants are linked.

        Variant mentions are linked if the variants they represent are biologically related. Two mentions are linked
        if any of the following conditions ar true:
        - `a` and `b` are equivalent variants on the same reference (e.g., same transcript)
        - `a` and `b` are equivalent variants on different references (e.g., transcript and genome)
        - `a` is the protein product of `b` or vice versa

        If either variant mention is invalid it can only be linked to another variant mention if they are equivalent.

        """
        # Handle equivalent variants.
        if a == b:
            return True

        # If either variant is invalid, they can only be linked if they're equivalent.
        if not a.valid or not b.valid:
            return False

        # If they're both in the same coordinate system (g, c, or p) and they're not equivalent, they aren't linked.
        if a.hgvs_desc[0] == b.hgvs_desc[0]:
            return False

        # If they're in different coordinate systems things get a little more complicated. We need to map and/or
        # translate to a common coordinate system to determine if they're linked.
        # Easiest thing to do is to map back to genomic coordinates and compare there.
        def c_to_g(c: str) -> List[str]:
            norm = mutalyzer_reference.normalize(c)
            if not norm or "chromosomal_descriptions" not in norm:
                return []
            for desc in norm["chromosomal_descriptions"]:
                if desc["assembly"] == "GRCH38":
                    return [desc["g"]]
            return []

        def p_to_g(p: str) -> List[str]:
            txs = mutalyzer_reference.back_translate(p)
            return [g for tx in txs for g in c_to_g(tx)]

        def map_to_g(v: HGVSVariant) -> List[str]:
            if v.hgvs_desc.startswith("g."):
                return [str(v)]
            elif v.hgvs_desc.startswith("c."):
                return c_to_g(str(v))
            elif v.hgvs_desc.startswith("p."):
                return p_to_g(str(v))
            else:
                return []

        # If the intersection of the genomic mappings is non-empty, the variants are linked.
        # TODO, handle parenthetical reference sequences and predicted descriptions.
        return bool(set(map_to_g(a)) & set(map_to_g(b)))

    def match(self, other: HGVSVariantMention) -> bool:
        """Determine if the new variant mention is a topical match for the topic."""
        for variant in self._variants:
            if self._linked(variant, other.variant):
                return True
        return False


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

# %%

variant_factory = HGVSVariantFactory()
variant_topics: List[VariantTopic] = []

for vt in variant_tuples:
    print(f"1: {vt}")
    try:
        v = variant_factory.try_parse(
            text_desc=vt["hgvs"] if vt["hgvs"] else vt["text"], gene_symbol=vt["gene"], refseq=vt["refseq"]
        )
    except ValueError as e:
        print(f"Error parsing variant {vt['text']}: {e}")
        continue
    print(f"2: {v}")
    vm = HGVSVariantMention(text=vt["text"], context="Some words", variant=v)
    print(f"3: {vm}")
    # Check all the topics for a match, if none, add a new topic.
    found = False
    for topic in variant_topics:
        if topic.match(vm):
            topic.add_mention(vm)
            found = True
            break
    if not found:
        variant_topics.append(VariantTopic([vm]))
    print(f"4: {variant_topics}")

# %% Some archival stuff

# TX_FASTA_REGEX = re.compile(r">(NM_\d+\.\d+) .*?\((.*?)\),")

# def get_mane_select_for_symbol(symbol: str) -> Tuple[str | None, int]:
#     """Fetch data on MANE select transcripts given a gene symbol.

#     Returns a tuple containing the MANE select transcript accession and the MANE select transcript ID
#     for a given gene symbol.
#     """
#     url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=nuccore&retmode=json&term={symbol}[Gene] AND refseq_select[filter] AND "Homo sapiens"[Organism]'
#     response = requests.get(url)
#     response.raise_for_status()

#     tx_entries = response.json()

#     if tx_entries["esearchresult"]["count"] == "0":
#         print(f"Warning ({symbol}): no MANE select transcript entries")
#         return (None, -1)
#     elif tx_entries["esearchresult"]["count"] != "1":
#         print(f"Warning ({symbol}): multiple MANE select entries ({tx_entries['esearchresult']['idlist']})")

#     for tx_id in tx_entries["esearchresult"]["idlist"]:

#         url2 = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&rettype=fasta&retmode=text&id={tx_id}"

#         response2 = requests.get(url2)
#         response2.raise_for_status()

#         tx_result = response2.text

#         rm = re.match(TX_FASTA_REGEX, tx_result)
#         if not rm:
#             print(f"Warning: FASTA response from nuccore for {tx_id} did not meet format expectations: {tx_result}")
#             continue

#         if rm.groups()[1] != symbol:
#             print(
#                 f"Warning: MANE select transcript {rm.groups()[0]} is not for gene {symbol} (it's for {rm.groups()[1]})"
#             )
#             continue

#         return (rm.groups()[0], tx_id)

#     return (None, -1)
