#!/usr/bin/env python3
"""PubTator3 Index Update Script - Downloads and updates PubTator3 indices.

This script checks for newer versions of PubTator3 data files and updates
the local indices if newer versions are available.

Usage:
    python pubtator_update_indices.py --index-dir /path/to/indices [--force]

Environment:
    PUBTATOR_INDEX_DIR: Default directory for index files
"""

import argparse
import gzip
import logging
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests


class PubTatorIndexUpdater:
    """Updates PubTator3 index files with download and validation."""

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.base_url = "https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator3/"
        self.gene_info_url = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/"
        self.logger = logging.getLogger(__name__)

        # Index files
        self.gene_disease_index = self.index_dir / "gene_disease_pmids.pkl"
        self.variant_pmids_index = self.index_dir / "variant_pmids.pkl"
        self.human_genes_index = self.index_dir / "human_genes.pkl"

        # Raw data files
        self.relation_file = self.index_dir / "relation2pubtator3.gz"
        self.variant_file = self.index_dir / "mutation2pubtator3.gz"
        self.gene_info_file = self.index_dir / "gene_info.gz"

        # Metadata file to track update times
        self.metadata_file = self.index_dir / ".pubtator_metadata.pkl"

    def check_for_updates(self) -> tuple[bool, list[str]]:
        """Check if any source files have newer versions available."""
        updates_needed = []

        # Check each source file
        files_to_check = [
            ("gene_info.gz", self.gene_info_url),
            ("relation2pubtator3.gz", self.base_url),
            ("mutation2pubtator3.gz", self.base_url),
        ]

        for filename, base_url in files_to_check:
            if self._is_update_available(filename, base_url):
                updates_needed.append(filename)

        return len(updates_needed) > 0, updates_needed

    def _is_update_available(self, filename: str, base_url: str) -> bool:
        """Check if a remote file is newer than the local version."""
        filepath = self.index_dir / filename

        # If file doesn't exist locally, update is needed
        if not filepath.exists():
            return True

        try:
            # Get remote file's last modified time
            response = requests.head(f"{base_url}{filename}")
            response.raise_for_status()

            remote_modified = response.headers.get("Last-Modified")
            if not remote_modified:
                # Can't determine, assume update needed
                return True

            # Parse remote modification time
            remote_time = datetime.strptime(remote_modified, "%a, %d %b %Y %H:%M:%S %Z")

            # Get local file modification time
            local_time = datetime.fromtimestamp(filepath.stat().st_mtime)

            # Check if remote is newer
            return remote_time > local_time

        except Exception as e:
            self.logger.warning(f"Could not check update status for {filename}: {e}")
            # On error, assume update might be needed
            return True

    def download_file(self, filename: str, base_url: str = None, force: bool = False):
        """Download file if it doesn't exist or if force is True."""
        if base_url is None:
            base_url = self.base_url

        filepath = self.index_dir / filename

        if filepath.exists() and not force:
            if not self._is_update_available(filename, base_url):
                self.logger.info(f"{filename} is up to date, skipping download")
                return
            else:
                self.logger.info(f"Newer version of {filename} available, downloading...")

        self.logger.info(f"Downloading {filename}...")
        try:
            response = requests.get(f"{base_url}{filename}", stream=True)
            response.raise_for_status()

            # Get file size for progress
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(
                            f"\rDownloading {filename}: {progress:.1f}%",
                            end="",
                            flush=True,
                        )

            print()  # New line after progress
            self.logger.info(f"Downloaded {filename}")
        except Exception as e:
            self.logger.error(f"Failed to download {filename}: {e}")
            raise

    def update_indices(self, force: bool = False):
        """Update all indices if newer versions are available."""
        # Check what needs updating
        if not force:
            needs_update, files_to_update = self.check_for_updates()
            if not needs_update:
                self.logger.info("All indices are up to date")
                return
            self.logger.info(f"Updates available for: {', '.join(files_to_update)}")

        # Download raw data files
        self.download_file("gene_info.gz", self.gene_info_url, force)
        self.download_file("relation2pubtator3.gz", force=force)
        self.download_file("mutation2pubtator3.gz", force=force)

        # Recreate all indices (since dependencies might have changed)
        self.logger.info("Recreating indices from updated data...")
        self._create_human_genes_index()
        self._create_gene_disease_index()
        self._create_variant_index()

        # Save metadata
        self._save_metadata()

        self.logger.info("Index update completed successfully")

    def _create_human_genes_index(self):
        """Create human genes index from NCBI gene_info file."""
        self.logger.info("Creating human genes index...")
        gene_id_to_symbol = {}
        symbol_to_gene_id = {}

        with gzip.open(self.gene_info_file, "rt") as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 1000000 == 0:
                    self.logger.debug(f"Processed {line_num:,} gene entries...")

                if line.startswith("#"):
                    continue

                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    tax_id = parts[0]
                    gene_id = parts[1]
                    symbol = parts[2]
                    synonyms = parts[4] if len(parts) > 4 else ""

                    # Only process human genes (taxid=9606)
                    if tax_id == "9606":
                        gene_id_int = int(gene_id)
                        gene_id_to_symbol[gene_id_int] = symbol
                        symbol_to_gene_id[symbol.upper()] = gene_id_int

                        # Also index synonyms
                        if synonyms and synonyms != "-":
                            for synonym in synonyms.split("|"):
                                if synonym.strip():
                                    symbol_to_gene_id[synonym.strip().upper()] = gene_id_int

        human_genes_data = {
            "gene_id_to_symbol": gene_id_to_symbol,
            "symbol_to_gene_id": symbol_to_gene_id,
            "human_gene_ids": set(gene_id_to_symbol.keys()),
        }

        with open(self.human_genes_index, "wb") as f:
            pickle.dump(human_genes_data, f)

        self.logger.info(f"Created human genes index with {len(gene_id_to_symbol):,} genes")

    def _create_gene_disease_index(self):
        """Create gene-disease association index from relations file."""
        # Load human genes data
        with open(self.human_genes_index, "rb") as f:
            human_genes_data = pickle.load(f)
        human_gene_ids = human_genes_data["human_gene_ids"]

        self.logger.info("Creating gene-disease index...")
        gene_disease_pmids = defaultdict(set)

        with gzip.open(self.relation_file, "rt") as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 1000000 == 0:
                    self.logger.debug(f"Processed {line_num:,} relations...")

                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    pmid_str = parts[0]
                    relation_type = parts[1]
                    entity1 = parts[2]
                    entity2 = parts[3]

                    try:
                        pmid = int(pmid_str)
                        if relation_type == "associate":
                            if entity1.startswith("Gene|") and entity2.startswith("Disease|"):
                                gene_id = int(entity1.replace("Gene|", ""))
                                if gene_id in human_gene_ids:
                                    gene_disease_pmids[gene_id].add(pmid)
                            elif entity1.startswith("Disease|") and entity2.startswith("Gene|"):
                                gene_id = int(entity2.replace("Gene|", ""))
                                if gene_id in human_gene_ids:
                                    gene_disease_pmids[gene_id].add(pmid)
                    except ValueError:
                        continue

        with open(self.gene_disease_index, "wb") as f:
            pickle.dump(gene_disease_pmids, f)

        self.logger.info(
            f"Created gene-disease index with {len(gene_disease_pmids):,} human genes having disease associations"
        )

    def _create_variant_index(self):
        """Create variant PMID index from mutation file."""
        self.logger.info("Creating variant index...")
        variant_pmids = set()

        with gzip.open(self.variant_file, "rt") as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 1000000 == 0:
                    self.logger.debug(f"Processed {line_num:,} variants...")

                pmid_str = line.split("\t")[0]
                try:
                    variant_pmids.add(int(pmid_str))
                except ValueError:
                    continue

        with open(self.variant_pmids_index, "wb") as f:
            pickle.dump(variant_pmids, f)

        self.logger.info(f"Created variant index with {len(variant_pmids):,} unique PMIDs")

    def _save_metadata(self):
        """Save metadata about the last update."""
        metadata = {
            "last_update": datetime.now(),
            "index_files": {
                "gene_disease_index": str(self.gene_disease_index),
                "variant_pmids_index": str(self.variant_pmids_index),
                "human_genes_index": str(self.human_genes_index),
            },
            "source_files": {
                "relation_file": str(self.relation_file),
                "variant_file": str(self.variant_file),
                "gene_info_file": str(self.gene_info_file),
            }
        }

        with open(self.metadata_file, "wb") as f:
            pickle.dump(metadata, f)


def main():
    """Main entry point for the update script."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Parse arguments
    parser = argparse.ArgumentParser(description="Update PubTator3 indices")
    parser.add_argument(
        "--index-dir",
        default=os.environ.get("PUBTATOR_INDEX_DIR", "."),
        help="Directory for index files (default: PUBTATOR_INDEX_DIR or current directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download and recreation of all indices",
    )

    args = parser.parse_args()

    # Initialize updater
    updater = PubTatorIndexUpdater(args.index_dir)

    # Run update
    try:
        updater.update_indices(force=args.force)
        logging.info("Update completed successfully")
    except Exception as e:
        logging.error(f"Update failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
