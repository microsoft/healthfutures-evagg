import gzip
import json
import logging
import os
import re
from typing import Any, Dict, Optional

import requests

from lib.evagg.utils import IWebContentClient

from .interfaces import IRefSeqLookupClient
from .ncbi import NcbiClientBase

logger = logging.getLogger(__name__)


class BaseLookupClient(NcbiClientBase, IRefSeqLookupClient):
    _DEFAULT_REFERENCE_DIR = ".ref"
    _ref: Dict[str, Dict[str, str]]
    _lazy_initialized: bool

    def __init__(
        self,
        web_client: IWebContentClient,
        settings: Optional[Dict[str, str]] = None,
        reference_dir: str = _DEFAULT_REFERENCE_DIR,
    ) -> None:
        # Lazy initialize so the constructor is fast.
        self._reference_dir = reference_dir
        self._lazy_initialized = False
        self._ref = {}

        super().__init__(web_client, settings)

    def _lazy_init(self) -> None:
        self._ref = {}
        self._lazy_initialized = True

    def transcript_accession_for_symbol(self, symbol: str) -> str | None:
        """Get the RefSeq transcript accession for a gene symbol."""
        if not self._lazy_initialized:
            self._lazy_init()
        return self._ref.get(symbol, {}).get("RNA", None)

    def protein_accession_for_symbol(self, symbol: str) -> str | None:
        """Get the RefSeq protein accession for a gene symbol."""
        if not self._lazy_initialized:
            self._lazy_init()
        return self._ref.get(symbol, {}).get("Protein", None)

    def genomic_accession_for_symbol(self, symbol: str) -> str | None:
        """Get the RefSeq genomic accession for a gene symbol."""
        if not self._lazy_initialized:
            self._lazy_init()
        return self._ref.get(symbol, {}).get("Genomic", None)

    def accession_autocomplete(self, accession: str) -> Optional[str]:
        """Get the latest RefSeq version for a versionless accession."""
        if accession.find(".") >= 0:
            logger.info(f"Accession '{accession}' is already versioned. Nothing to do.")
            return accession

        result = self._efetch(db="nuccore", id=accession, retmode="text", rettype="acc")
        result = result.strip()

        if result.startswith("Error") is False:
            return result
        return None

    def _download_reference(self, url: str, target: str) -> None:
        # Download a reference TSV from `url`.
        # Note, this fail if a cosmos caching web client is provided and a large reference is being downloaded.
        with open(target, "w") as f:
            f.write(self._web_client.get(url=url))


class RefSeqLookupClient(BaseLookupClient):
    """Determine RefSeq 'Mane Select' and 'RefSeq Select' accessions for genes using the NCBI RefSeq database."""

    _NCBI_REFSEQ_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/vertebrate_mammalian/Homo_sapiens/all_assembly_versions/GCF_000001405.39_GRCh38.p13/GCF_000001405.39_GRCh38.p13_genomic.gff.gz"  # noqa: E501
    _RAW_FILENAME = "GCF_000001405.39_GRCh38.p13_genomic.gff.gz"
    _PROCESSED_FILEPATH = "refseq_processed.json"

    def _download_binary_reference(self, url: str, target: str) -> None:
        response = requests.get(url)
        response.raise_for_status()

        with open(target, "wb") as f:
            f.write(response.content)

    def _process_raw_resource(self) -> Dict[str, Any]:
        raw_target_filepath = os.path.join(self._reference_dir, self._RAW_FILENAME)

        if not os.path.exists(raw_target_filepath):
            logger.info(f"Downloading reference file to {raw_target_filepath}")
            print(f"Downloading reference file to {raw_target_filepath}")

            self._download_binary_reference(self._NCBI_REFSEQ_URL, raw_target_filepath)

        protein_lines = []
        transcript_lines = []

        refseqs: Dict[str, Any] = {}

        logger.info("Processing raw reference file.")
        with gzip.open(raw_target_filepath, "rt") as f:
            for line in f:
                if not re.search(r"(MANE|RefSeq) Select", line):
                    continue
                if re.search(r"BestRefSeq\s+CDS", line):
                    protein_lines.append(line)
                elif re.search(r"BestRefSeq\s+mRNA", line):
                    transcript_lines.append(line)

        os.remove(raw_target_filepath)

        for line in protein_lines:
            tokens = line.split("\t")
            assert len(tokens) == 9, "Wrong number of tokens in line"
            attributes = {kv.split("=")[0]: kv.split("=")[1] for kv in tokens[8].split(";")}
            assert "gene" in attributes, "gene not found in attributes"
            assert "tag" in attributes and "Select" in attributes["tag"], "Not tagged as select"
            assert "protein_id" in attributes, "protein_id not found in attributes"
            assert "Dbxref" in attributes, "Dbxref not found in attributes"
            xref = {kv.split(":")[0]: kv.split(":")[1] for kv in attributes["Dbxref"].split(",")}
            assert "GeneID" in xref, "GeneID not found in DBxref"

            if attributes["gene"] in refseqs:
                if refseqs[attributes["gene"]]["MANE"]:
                    continue
                elif "MANE Select" in attributes["tag"]:
                    print(f"Warning: {attributes['gene']} already has a non-MANE protein, replacing.")

            refseqs[attributes["gene"]] = {
                "Protein": attributes["protein_id"],
                "Genomic": tokens[0],
                "Symbol": attributes["gene"],
                "GeneID": xref["GeneID"],
                "MANE": "MANE Select" in attributes["tag"],
            }

        for line in transcript_lines:
            tokens = line.split("\t")
            assert len(tokens) == 9, "Wrong number of tokens in line"
            attributes = {kv.split("=")[0]: kv.split("=")[1] for kv in tokens[8].split(";")}
            assert "gene" in attributes, "gene not found in attributes"
            assert "tag" in attributes and "Select" in attributes["tag"], "Not tagged as Select"
            assert "transcript_id" in attributes, "transcript_id not found in attributes"

            if attributes["gene"] not in refseqs:
                print(f"Warning: {attributes['gene']} not found in proteins")
                continue

            if "RNA" in refseqs[attributes["gene"]]:
                print(f"Warning: {attributes['gene']} already has an RNA")
                continue

            if "MANE Select" in attributes["tag"] and not refseqs[attributes["gene"]]["MANE"]:
                print(f"Warning: {attributes['gene']} has a non-MANE protein, but a MANE RNA.")

            refseqs[attributes["gene"]]["RNA"] = attributes["transcript_id"].strip()

        logger.info(f"Processed {len(refseqs)} RefSeq entries.")

        return refseqs

    def _lazy_init(self) -> None:
        # Download the reference file if necessary.
        if not os.path.exists(self._reference_dir):
            logging.info(f"Creating reference directory at {self._reference_dir}")
            os.makedirs(self._reference_dir)

        resource_path = os.path.join(self._reference_dir, self._PROCESSED_FILEPATH)

        if not os.path.exists(resource_path):
            self._ref = self._process_raw_resource()
            json.dump(self._ref, open(resource_path, "w"), indent=4)
        else:
            self._ref = json.load(open(resource_path, "r"))

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
        field_mapping = {"RSG": "Genomic"}

        reference_dict = {}
        for line in lines[1:]:
            fields = line.strip().split("\t")
            if fields[-1] != "reference standard":
                continue
            gene_symbol = fields[2]
            if gene_symbol in reference_dict:
                logging.debug(f"Multiple reference standard entries for gene {gene_symbol}. Keeping the first one.")
                continue
            reference_dict[gene_symbol] = {
                field_mapping.get(k, k): v for k, v in zip(header, fields) if k in kept_fields
            }

        return reference_dict


class RefSeqGeneLookupClient(BaseLookupClient):
    """Determine RefSeq 'Reference Standard' accessions for genes using the NCBI RefSeqGene database."""

    _NCBI_REFSEQGENE_URL = "https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/RefSeqGene/LRG_RefSeqGene"

    _MANUAL_ADDITIONS = {
        "DNAJC7": {  # No RefSeqGene accession, thus not in reference table. Using chromosomal genomic reference.
            "GeneID": "7266",
            "Symbol": "DNAJC7",
            "Genomic": "NC_000017.11",
            "RNA": "NM_003315.4",
            "Protein": "NP_003306.3",
        },
        "SARS1": {  # No RefSeqGene accession, thus not in reference table. Using chromosomal genomic reference.
            "GeneID": "6301",
            "Symbol": "SARS1",
            "Genomic": "NC_000001.11",
            "RNA": "NM_006513.4",
            "Protein": "NP_006504.2",
        },
        "LRRC10": {  # No RefSeqGene accession, thus not in reference table. Using chromosomal genomic reference.
            "GeneID": "23568",
            "Symbol": "LRRC10",
            "Genomic": "NC_000012.12",
            "RNA": "NM_201550.4",
            "Protein": "NP_963844.2",
        },
        "TOPBP1": {  # No RefSeqGene accession, thus not in reference table. Using chromosomal genomic reference.
            "GeneID": "11073",
            "Symbol": "TOPBP1",
            "Genomic": "NC_000003.12",
            "RNA": "NM_007027.4",
            "Protein": "NP_008958.2",
        },
        "TDO2": {  # No RefSeqGene accession, thus not in reference table. Using chromosomal genomic reference.
            "GeneID": "6999",
            "Symbol": "TDO2",
            "Genomic": "NC_000004.12",
            "RNA": "NM_005651.4",
            "Protein": "NP_005642.1",
        },
        "ADGRD1": {  # No RefSeqGene accession, thus not in reference table. Using chromosomal genomic reference.
            "GeneID": "283383",
            "Symbol": "ADGRD1",
            "Genomic": "NC_000012.12",
            "RNA": "NM_198827.5",
            "Protein": "NP_942122.2",
        },
        "ALPK2": {  # No RefSeqGene accession, thus not in reference table. Using chromosomal genomic reference.
            "GeneID": "115701",
            "Symbol": "ALPK2",
            "Genomic": "NC_000018.10",
            "RNA": "NM_052947.4",
            "Protein": "NP_443179.3",
        },
        "ANKS3": {  # No RefSeqGene accession, thus not in reference table. Using chromosomal genomic reference.
            "GeneID": "124401",
            "Symbol": "ANKS3",
            "Genomic": "NC_000016.10",
            "RNA": "NM_133450.4",
            "Protein": "NP_597707.1",
        },
        "BAHCC1": {  # No RefSeqGene accession, thus not in reference table. Using chromosomal genomic reference.
            "GeneID": "57597",
            "Symbol": "BAHCC1",
            "Genomic": "NC_000017.11",
            "RNA": "NM_001377448.1",
            "Protein": "NP_001364377.1",
        },
        "BOD1": {  # No RefSeqGene accession, thus not in reference table. Using chromosomal genomic reference.
            "GeneID": "91272",
            "Symbol": "BOD1",
            "Genomic": "NC_000005.10",
            "RNA": "NM_138369.3",
            "Protein": "NP_612378.1",
        },
        "CARF": {  # No RefSeqGene accession, thus not in reference table. Using chromosomal genomic reference.
            "GeneID": "79800",
            "Symbol": "CARF",
            "Genomic": "NC_000002.12",
            "RNA": "NM_024744.17",
            "Protein": "NP_079020.13",
        },
        "CASZ1": {  # No RefSeqGene accession, thus not in reference table. Using chromosomal genomic reference.
            "GeneID": "54897",
            "Symbol": "CASZ1",
            "Genomic": "NC_000001.11",
            "RNA": "NM_001079843.3",
            "Protein": "NP_001073312.1",
        },
    }

    def _lazy_init(self) -> None:
        # Download the reference file if necessary.
        if not os.path.exists(self._reference_dir):
            logging.info(f"Creating reference directory at {self._reference_dir}")
            os.makedirs(self._reference_dir)

        reference_filepath = os.path.join(self._reference_dir, "LRG_RefSeqGene.tsv")
        if not os.path.exists(reference_filepath):
            logging.info(f"Downloading reference file to {reference_filepath}")
            self._download_reference(self._NCBI_REFSEQGENE_URL, reference_filepath)

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
        field_mapping = {"RSG": "Genomic"}

        reference_dict = {}
        for line in lines[1:]:
            fields = line.strip().split("\t")
            if fields[-1] != "reference standard":
                continue
            gene_symbol = fields[2]
            if gene_symbol in reference_dict:
                logging.debug(f"Multiple reference standard entries for gene {gene_symbol}. Keeping the first one.")
                continue
            reference_dict[gene_symbol] = {
                field_mapping.get(k, k): v for k, v in zip(header, fields) if k in kept_fields
            }

        return reference_dict
