import logging
import os
from typing import Dict, Optional

from lib.evagg.utils import IWebContentClient

from .interfaces import IRefSeqLookupClient
from .ncbi import NcbiClientBase

logger = logging.getLogger(__name__)


class NcbiReferenceLookupClient(NcbiClientBase, IRefSeqLookupClient):
    """Determine RefSeq 'Reference Standard' accessions for genes using the NCBI RefSeqGene database."""

    _ref: Dict[str, Dict[str, str]]
    _lazy_initialized: bool
    _DEFAULT_REFERENCE_DIR = ".ref"
    _NCBI_REFERENCE_URL = "https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/RefSeqGene/LRG_RefSeqGene"

    _MANUAL_ADDITIONS = {
        "DNAJC7": {  # No RefSeqGene accession, thus not in reference table. Using chromosomal genomic reference.
            "GeneID": "7266",
            "Symbol": "DNAJC7",
            "RSG": "NC_000017.11",
            "RNA": "NM_003315.4",
            "Protein": "NP_003306.3",
        },
        "SARS1": {  # No RefSeqGene accession, thus not in reference table. Using chromosomal genomic reference.
            "GeneID": "6301",
            "Symbol": "SARS1",
            "RSG": "NC_000001.11",
            "RNA": "NM_006513.4",
            "Protein": "NP_006504.2",
        },
        "LRRC10": {  # No RefSeqGene accession, thus not in reference table. Using chromosomal genomic reference.
            "GeneID": "23568",
            "Symbol": "LRRC10",
            "RSG": "NC_000012.12",
            "RNA": "NM_201550.4",
            "Protein": "NP_963844.2",
        },
        "TOPBP1": {  # No RefSeqGene accession, thus not in reference table. Using chromosomal genomic reference.
            "GeneID": "11073",
            "Symbol": "TOPBP1",
            "RSG": "NC_000003.12",
            "RNA": "NM_007027.4",
            "Protein": "NP_008958.2",
        },
    }

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
        return self._ref.get(symbol, {}).get("RSG", None)

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
        # Download the reference TSV file from NCBI.
        # TODO: this will currently fail if a caching web client is provided due to the size of the reference.
        with open(target, "w") as f:
            f.write(self._web_client.get(url=url))

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
                logging.debug(f"Multiple reference standard entries for gene {gene_symbol}. Keeping the first one.")
                continue
            reference_dict[gene_symbol] = {k: v for k, v in zip(header, fields) if k in kept_fields}

        # The reference has known missing genes. I hate to do this, but we're going to manually add them in here.
        for k, v in self._MANUAL_ADDITIONS.items():
            if k in reference_dict:
                # If the gene is already in the reference table, don't overwrite it.
                continue
            reference_dict[k] = v

        return reference_dict
