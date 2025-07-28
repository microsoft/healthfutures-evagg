#!/usr/bin/env python3
"""PubTator3 MCP Server - Search gene-variant-disease associations and extract variants.

This MCP server provides tools to:
1. Search PubTator3 data for papers with gene-disease associations and variants
2. Extract genetic variants from biomedical text using tmVar3

Usage:
    python pubtator_mcp_server.py --index-dir /path/to/indices --tmvar-dir /path/to/tmVar3

Environment:
    PUBTATOR_INDEX_DIR: Default directory for index files
    TMVAR_DIR: Default directory for tmVar3 installation

Note: Index files must be created first using pubtator_update_indices.py
Note: tmVar3 must be installed for variant extraction functionality
"""

import argparse
import asyncio
import logging
import os
import pickle
import re
import shutil
import sys
import uuid
from pathlib import Path

import jdk4py
from fastmcp import FastMCP

from .models import (
    ErrorResult,
    GeneNormalizationResult,
    GeneVariantSearchRequest,
    GeneVariantSearchResult,
    VariantExtractionRequest,
    VariantResult,
)

logger = logging.getLogger(__name__)


def _extract_document_id(bioc_xml: str) -> str:
    """Extract document ID from BioC XML."""
    # Look for <id>document_id</id> pattern in the XML
    match = re.search(r"<id>\s*([^<]+)\s*</id>", bioc_xml)
    if not match:
        raise ValueError("No document ID found in BioC XML. Expected <id>document_id</id> element.")

    doc_id = match.group(1).strip()
    if not doc_id:
        raise ValueError("Empty document ID found in BioC XML.")

    # Sanitize the ID to be filesystem-safe
    doc_id = re.sub(r"[^\w\-_.]", "_", doc_id)
    return doc_id


class PubTatorIndexManager:
    """Manages PubTator3 index file validation."""

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.logger = logging.getLogger(__name__)

        # Index files
        self.gene_disease_index = self.index_dir / "gene_disease_pmids.pkl"
        self.variant_pmids_index = self.index_dir / "variant_pmids.pkl"
        self.human_genes_index = self.index_dir / "human_genes.pkl"

    def check_indices(self) -> tuple[bool, list[str]]:
        """Check if all required indices exist."""
        missing = []
        required_files = [
            self.gene_disease_index,
            self.variant_pmids_index,
            self.human_genes_index,
        ]

        for file in required_files:
            if not file.exists():
                missing.append(file.name)

        return len(missing) == 0, missing


class PubTatorSearcher:
    """Performs searches on PubTator3 indices."""

    def __init__(self, index_manager: PubTatorIndexManager):
        self.index_manager = index_manager
        self.logger = logging.getLogger(__name__)
        self._indices_loaded = False
        self.gene_disease_pmids = None
        self.variant_pmids = None
        self.human_genes_data = None
        self._gene_cache = {}  # symbol -> GeneNormalizationResult

    def load_indices(self):
        """Load indices into memory."""
        if self._indices_loaded:
            return

        self.logger.info("Loading indices...")

        with open(self.index_manager.gene_disease_index, "rb") as f:
            self.gene_disease_pmids = pickle.load(f)

        with open(self.index_manager.variant_pmids_index, "rb") as f:
            self.variant_pmids = pickle.load(f)

        with open(self.index_manager.human_genes_index, "rb") as f:
            self.human_genes_data = pickle.load(f)

        self._indices_loaded = True
        self.logger.info("Indices loaded successfully")

    def normalize_gene_symbol(self, gene_symbol: str) -> GeneNormalizationResult:
        """Normalize gene symbol to NCBI Gene ID."""
        # Check cache first
        if gene_symbol in self._gene_cache:
            return self._gene_cache[gene_symbol]

        self.load_indices()

        symbol_to_gene_id = self.human_genes_data["symbol_to_gene_id"]
        gene_id_to_symbol = self.human_genes_data["gene_id_to_symbol"]

        # Look up symbol (case-insensitive)
        gene_id = symbol_to_gene_id.get(gene_symbol.upper())

        if gene_id:
            official_symbol = gene_id_to_symbol.get(gene_id, gene_symbol)
            result = GeneNormalizationResult(
                input_symbol=gene_symbol,
                gene_id=gene_id,
                official_symbol=official_symbol,
                found=True,
            )
        else:
            result = GeneNormalizationResult(
                input_symbol=gene_symbol,
                gene_id=None,
                official_symbol=None,
                found=False,
                error=f"No human gene found for symbol '{gene_symbol}'",
            )

        # Cache the result
        self._gene_cache[gene_symbol] = result
        return result

    def search_gene_variant_papers(
        self,
        request: GeneVariantSearchRequest,
    ) -> GeneVariantSearchResult | ErrorResult:
        """Find papers that mention both gene-disease associations and variants."""
        self.load_indices()

        # Normalize gene symbol
        norm_result = self.normalize_gene_symbol(request.gene_symbol)
        if not norm_result.found:
            return ErrorResult(error=norm_result.error or "Gene normalization failed")

        gene_id = norm_result.gene_id

        # Get PMIDs for this gene's disease associations
        gene_disease_papers = self.gene_disease_pmids.get(gene_id, set())

        # Find intersection with variant PMIDs
        intersection_pmids = gene_disease_papers.intersection(self.variant_pmids)

        # Sort PMIDs in reverse numerical order (newest PMIDs first)
        sorted_pmids = sorted(intersection_pmids, reverse=True)
        full_count = len(sorted_pmids)

        # Apply retmax limit if specified
        if request.retmax is not None and request.retmax > 0:
            sorted_pmids = sorted_pmids[: request.retmax]

        # TODO: Implement date filtering when min_date/max_date are provided
        # This would require fetching publication dates for PMIDs

        return GeneVariantSearchResult(
            gene_symbol=norm_result.official_symbol or request.gene_symbol,
            full_count=full_count,
            pmids=sorted_pmids,
        )


class TmVarProcessor:
    """Processes variant extraction using tmVar3."""

    def __init__(self, tmvar_dir: Path, max_concurrent: int = 3) -> None:
        """Initialize tmVar processor.

        Args:
            tmvar_dir: Path to tmVar3 installation directory
            max_concurrent: Maximum number of concurrent tmVar3 processes
        """
        self.tmvar_dir = tmvar_dir
        self.logger = logging.getLogger(__name__)
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Validate tmVar3 installation
        self._validate_installation()

    def _validate_installation(self) -> None:
        """Validate that tmVar3 is properly installed."""
        required_files = [
            self.tmvar_dir / "tmVar.jar",
            self.tmvar_dir / "lib" / "sqlite-jdbc-3.49.1.0.jar",
            self.tmvar_dir / "CRF" / "crf_test",
        ]

        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required tmVar3 file not found: {file_path}")

        self.logger.info(f"tmVar3 installation validated at {self.tmvar_dir}")

    async def extract_variants(
        self, request: VariantExtractionRequest, searcher: PubTatorSearcher | None = None
    ) -> list[VariantResult] | ErrorResult:
        """Extract variants from text using tmVar3."""
        try:
            async with self._semaphore:
                doc_id = _extract_document_id(request.text)
                output_text = await self._run_tmvar(request.text, doc_id)
                variants = self._parse_pubtator_output(output_text, request.gene_symbol, searcher)
                return variants
        except Exception as e:
            self.logger.error(f"Variant extraction failed: {e}")
            return ErrorResult(error=f"Variant extraction failed: {str(e)}")

    async def _run_tmvar(self, text: str, doc_id: str) -> str:
        """Run tmVar3 process with unique temporary directories."""
        # Write input file - expect BioC XML format
        if not (text.strip().startswith("<?xml") or text.strip().startswith("<")):
            raise ValueError("tmVar3 requires BioC XML format input, not plain text")

        # Create unique temporary directories for this invocation
        unique_id = str(uuid.uuid4())
        temp_base = self.tmvar_dir / f"temp_{unique_id}"

        try:
            # Create temporary input and output directories
            input_dir = temp_base / "input"
            output_dir = temp_base / "output"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Write input file
            input_file = input_dir / f"{doc_id}.txt"
            input_file.write_text(text, encoding="utf-8")

            # Run tmVar3
            await self._execute_java_process(input_dir, output_dir)

            # Read output
            output_pubtator = output_dir / f"{doc_id}.txt.PubTator"
            if output_pubtator.exists():
                result = output_pubtator.read_text(encoding="utf-8")
            else:
                raise RuntimeError("tmVar3 did not produce expected output file")

            return result

        finally:
            # Clean up the entire temporary directory tree
            if temp_base.exists():
                shutil.rmtree(temp_base)
                self.logger.debug(f"Cleaned up temporary directory: {temp_base}")

    async def _execute_java_process(self, input_dir: Path, output_dir: Path) -> None:
        """Execute tmVar3 Java process."""
        # Make paths relative to tmvar_dir since we execute with cwd=self.tmvar_dir
        relative_input_dir = input_dir.relative_to(self.tmvar_dir)
        relative_output_dir = output_dir.relative_to(self.tmvar_dir)

        # Java arguments for memory and performance
        java_cmd = [
            str(jdk4py.JAVA),
            "-XX:ActiveProcessorCount=2",
            "-Xmx10G",
            "-Xms5G",
            "-cp",
            "lib/sqlite-jdbc-3.49.1.0.jar:tmVar.jar",
            "org.eclipse.jdt.internal.jarinjarloader.JarRsrcLoader",
            str(relative_input_dir),
            str(relative_output_dir),
        ]

        self.logger.info(f"Executing tmVar3: {' '.join(java_cmd)}")

        # Execute process (must run in tmvar_dir so ./CRF/crf_test is found)
        try:
            process = await asyncio.create_subprocess_exec(
                *java_cmd,
                cwd=self.tmvar_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = f"tmVar3 failed with return code {process.returncode}"
                if stderr:
                    error_msg += f": {stderr.decode('utf-8', errors='ignore')}"
                raise RuntimeError(error_msg)

            self.logger.info("tmVar3 execution completed successfully")

        except Exception as e:
            self.logger.error(f"Failed to execute tmVar3: {e}")
            raise RuntimeError(f"tmVar3 execution failed: {e}") from e

    def _parse_pubtator_output(
        self, output_text: str, gene_filter: str | None, searcher: PubTatorSearcher | None
    ) -> list[VariantResult]:
        """Parse tmVar3 PubTator output to extract variants."""
        variants = []

        # Pre-compute filter normalization once instead of per variant
        filter_norm = None
        if gene_filter and searcher:
            filter_norm = searcher.normalize_gene_symbol(gene_filter)

        for line in output_text.strip().split("\n"):
            if not line or line.startswith("#"):
                continue

            # PubTator format: PMID\tstart\tend\ttext\ttype\tnormalization
            parts = line.split("\t")
            if len(parts) >= 5:
                try:
                    # Filter for variant types only, skip table and other non-variant data
                    variant_type = parts[4]
                    if variant_type not in ["DNAMutation", "ProteinMutation", "SNP", "Mutation", "Variation"]:
                        continue

                    start_pos = int(parts[1])
                    end_pos = int(parts[2])
                    variant_text = parts[3]
                    normalization = parts[5] if len(parts) > 5 else ""

                    # Parse normalization data
                    ncbi_gene_id, gene_symbol, hgvs, rs_id = self._parse_normalization(normalization)

                    # Apply gene filter if specified
                    if gene_filter and gene_symbol:
                        # Use pre-computed filter normalization
                        if searcher and filter_norm:
                            variant_norm = searcher.normalize_gene_symbol(gene_symbol)
                            if filter_norm.found and variant_norm.found and filter_norm.gene_id != variant_norm.gene_id:
                                continue
                        elif gene_symbol.upper() != gene_filter.upper():
                            continue

                    variants.append(
                        VariantResult(
                            text=variant_text,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            ncbi_gene_id=ncbi_gene_id,
                            gene_symbol=gene_symbol,
                            hgvs=hgvs,
                            rs_id=rs_id,
                        )
                    )
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Failed to parse line: {line} - {e}")
                    continue

        return variants

    def _parse_normalization(self, normalization: str) -> tuple[int | None, str | None, str | None, str | None]:
        """Parse tmVar3 normalization string."""
        ncbi_gene_id = None
        gene_symbol = None
        hgvs = None
        rs_id = None

        if not normalization:
            return ncbi_gene_id, gene_symbol, hgvs, rs_id

        # Parse fields like: CorrespondingGene:7139;RS#:397516456;HGVS:p.R92W
        for field in normalization.split(";"):
            if ":" in field:
                key, value = field.split(":", 1)
                if key == "CorrespondingGene" and value.isdigit():
                    ncbi_gene_id = int(value)
                elif key == "HGVS":
                    hgvs = value
                elif key == "RS#":
                    rs_id = value

        return ncbi_gene_id, gene_symbol, hgvs, rs_id


class GNorm2Processor:
    """Processes gene normalization using GNorm2."""

    def __init__(self, gnorm2_dir: Path, max_concurrent: int = 2) -> None:
        """Initialize GNorm2 processor.

        Args:
            gnorm2_dir: Path to GNorm2 installation directory
            max_concurrent: Maximum number of concurrent GNorm2 processes
        """
        self.gnorm2_dir = gnorm2_dir
        self.logger = logging.getLogger(__name__)
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Validate GNorm2 installation
        self._validate_installation()

    def _validate_installation(self) -> None:
        """Validate that GNorm2 is properly installed."""
        required_files = [
            self.gnorm2_dir / "GNormPlus.jar",
            self.gnorm2_dir / "GeneNER_SpeAss_run.py",
            self.gnorm2_dir / ".venv" / "pyvenv.cfg",
            self.gnorm2_dir / "gnorm_trained_models" / "GeneNER" / "GeneNER-Bioformer-BEST.h5",
            self.gnorm2_dir / "gnorm_trained_models" / "SpeAss" / "SpeAss-Bioformer-SG-BEST.h5",
        ]

        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required GNorm2 file not found: {file_path}")

        self.logger.info(f"GNorm2 installation validated at {self.gnorm2_dir}")

    async def normalize_genes(self, bioc_xml: str, doc_id: str) -> str:
        """Run GNorm2 gene normalization on BioC XML text."""
        try:
            async with self._semaphore:
                return await self._run_gnorm2_pipeline(bioc_xml, doc_id)
        except Exception as e:
            self.logger.error(f"GNorm2 gene normalization failed: {e}")
            raise RuntimeError(f"GNorm2 normalization failed: {str(e)}") from e

    async def _run_gnorm2_pipeline(self, bioc_xml: str, doc_id: str) -> str:
        """Run the full GNorm2 pipeline with unique temporary directories."""
        # Create unique temporary directories for this invocation
        unique_id = str(uuid.uuid4())
        temp_base = self.gnorm2_dir / f"temp_{unique_id}"

        try:
            # Create all temporary directories
            input_dir = temp_base / "input"
            tmp_sr_dir = temp_base / "tmp_SR"
            tmp_gnr_dir = temp_base / "tmp_GNR"
            tmp_sa_dir = temp_base / "tmp_SA"
            output_dir = temp_base / "output"

            # Ensure directories exist
            for dir_path in [input_dir, tmp_sr_dir, tmp_gnr_dir, tmp_sa_dir, output_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)

            # Write input BioC XML file using doc_id
            input_file = input_dir / f"{doc_id}.txt"
            input_file.write_text(bioc_xml, encoding="utf-8")

            # Step 1: Species Recognition
            await self._run_java_step(temp_base / "input", temp_base / "tmp_SR", "setup.SR.txt", "Species Recognition")

            # Step 2: Species Assignment + Gene Name Recognition
            await self._run_python_step(
                temp_base / "tmp_SR",
                temp_base / "tmp_GNR",
                temp_base / "tmp_SA",
                "Species Assignment + Gene Name Recognition",
            )

            # Step 3: Gene Normalization
            await self._run_java_step(temp_base / "tmp_SA", temp_base / "output", "setup.GN.txt", "Gene Normalization")

            # Read the final output
            output_file = output_dir / f"{doc_id}.txt"
            if output_file.exists():
                result = output_file.read_text(encoding="utf-8")
            else:
                raise RuntimeError("GNorm2 did not produce expected output file")

            return result

        finally:
            # Clean up the entire temporary directory tree
            if temp_base.exists():
                shutil.rmtree(temp_base)
                self.logger.debug(f"Cleaned up temporary directory: {temp_base}")

    async def _run_java_step(self, input_dir: Path, output_dir: Path, setup_file: str, step_name: str) -> None:
        """Execute a GNorm2 Java step."""
        # Make paths relative to gnorm2_dir for the Java command
        input_subdir = str(input_dir.relative_to(self.gnorm2_dir))
        output_subdir = str(output_dir.relative_to(self.gnorm2_dir))

        java_cmd = [
            str(jdk4py.JAVA),
            "-Xmx10G",
            "-Xms5G",
            "-jar",
            "GNormPlus.jar",
            input_subdir,
            output_subdir,
            setup_file,
        ]

        self.logger.info(f"Executing GNorm2 {step_name}: {' '.join(java_cmd)}")

        try:
            process = await asyncio.create_subprocess_exec(
                *java_cmd,
                cwd=self.gnorm2_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = f"GNorm2 {step_name} failed with return code {process.returncode}"
                if stderr:
                    error_msg += f": {stderr.decode('utf-8', errors='ignore')}"
                raise RuntimeError(error_msg)

            self.logger.info(f"GNorm2 {step_name} completed successfully")

        except Exception as e:
            self.logger.error(f"Failed to execute GNorm2 {step_name}: {e}")
            raise RuntimeError(f"GNorm2 {step_name} execution failed: {e}") from e

    async def _run_python_step(
        self, input_dir: Path, ner_output_dir: Path, sa_output_dir: Path, step_name: str
    ) -> None:
        """Execute the GNorm2 Python in the virtual environent with the GNorm2 dependencies installed."""
        # Make paths relative to gnorm2_dir for the Python command
        input_subdir = str(input_dir.relative_to(self.gnorm2_dir))
        ner_output_subdir = str(ner_output_dir.relative_to(self.gnorm2_dir))
        sa_output_subdir = str(sa_output_dir.relative_to(self.gnorm2_dir))

        python_cmd = [
            ".venv/bin/python",
            "GeneNER_SpeAss_run.py",
            "-i",
            input_subdir,
            "-r",
            ner_output_subdir,
            "-a",
            sa_output_subdir,
            "-n",
            "gnorm_trained_models/GeneNER/GeneNER-Bioformer-BEST.h5",
            "-s",
            "gnorm_trained_models/SpeAss/SpeAss-Bioformer-SG-BEST.h5",
        ]

        self.logger.info(f"Executing GNorm2 {step_name}: {' '.join(python_cmd)}")

        try:
            process = await asyncio.create_subprocess_exec(
                *python_cmd,
                cwd=self.gnorm2_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = f"GNorm2 {step_name} failed with return code {process.returncode}"
                if stderr:
                    error_msg += f": {stderr.decode('utf-8', errors='ignore')}"
                raise RuntimeError(error_msg)

            self.logger.info(f"GNorm2 {step_name} completed successfully")

        except Exception as e:
            self.logger.error(f"Failed to execute GNorm2 {step_name}: {e}")
            raise RuntimeError(f"GNorm2 {step_name} execution failed: {e}") from e


# Create the MCP server
mcp = FastMCP("PubTator MCP")

# Global instances (will be initialized in main)
index_manager: PubTatorIndexManager | None = None
searcher: PubTatorSearcher | None = None
gnorm2_processor: GNorm2Processor | None = None
tmvar_processor: TmVarProcessor | None = None


@mcp.tool
def gene_symbol_to_ncbi_id(gene_symbol: str) -> GeneNormalizationResult | ErrorResult:
    """Convert a gene symbol to its NCBI Gene ID.

    This tool normalizes gene symbols (including synonyms) to their official
    NCBI Gene IDs for human genes.
    """
    logger.debug("Invoked MCP tool: gene_symbol_to_ncbi_id")

    if not searcher:
        return ErrorResult(error="Server not properly initialized. Index files may be missing.")

    try:
        return searcher.normalize_gene_symbol(gene_symbol)
    except Exception as e:
        logger.error(f"Error normalizing gene symbol: {e}")
        return ErrorResult(error=f"Failed to normalize gene symbol: {str(e)}")


@mcp.tool
def search_gene_variant_papers(request: GeneVariantSearchRequest) -> GeneVariantSearchResult | ErrorResult:
    """Find papers that describe gene-disease associations and also mention variants."""
    logger.debug("Invoked MCP tool: search_gene_variant_papers")

    if not searcher:
        return ErrorResult(error="Server not properly initialized. Index files may be missing.")

    try:
        return searcher.search_gene_variant_papers(request)
    except Exception as e:
        logger.error(f"Error searching gene variant papers: {e}")
        return ErrorResult(error=f"Search failed: {str(e)}")


@mcp.tool
async def extract_variants(request: VariantExtractionRequest) -> list[VariantResult] | ErrorResult:
    """Extract genetic variants from biomedical text using GNorm2 + tmVar3.

    This tool first uses GNorm2 to normalize genes in the text, then uses tmVar3
    to identify and extract genetic variant mentions. The text must be in BioC
    XML format - plain text is not supported. If gene_symbol is provided, only
    variants associated with that gene will be returned.
    """
    logger.debug("Invoked MCP tool: extract_variants")

    if not gnorm2_processor:
        return ErrorResult(error="GNorm2 not available. Please ensure GNorm2 is properly installed.")

    if not tmvar_processor:
        return ErrorResult(error="tmVar3 not available. Please ensure tmVar3 is properly installed.")

    try:
        # Preprocess text: replace tabs with spaces to avoid tmVar confusion
        preprocessed_text = request.text.replace("\t", " ")

        # Extract document ID from the BioC XML
        doc_id = _extract_document_id(preprocessed_text)

        # Step 1: Run GNorm2 gene normalization
        gnorm2_output = await gnorm2_processor.normalize_genes(preprocessed_text, doc_id)

        # Step 2: Run tmVar3 on the GNorm2-processed text
        tmvar_request = VariantExtractionRequest(text=gnorm2_output, gene_symbol=request.gene_symbol)
        return await tmvar_processor.extract_variants(tmvar_request, searcher)
    except Exception as e:
        logger.error(f"Error extracting variants: {e}")
        return ErrorResult(error=f"Failed to extract variants: {str(e)}")


def main():
    """Main entry point for the MCP server."""
    global index_manager, searcher, gnorm2_processor, tmvar_processor

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Parse arguments
    parser = argparse.ArgumentParser(description="PubTator MCP Server")
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("PUBTATOR_CACHE_DIR", "../../.cache"),
        help=(
            "Cache directory containing pubtator_search_index and tmVar3 subdirs "
            "(default: PUBTATOR_CACHE_DIR or ../../.cache)"
        ),
    )

    args = parser.parse_args()

    # Set up directories
    cache_dir = Path(args.cache_dir)
    index_dir = cache_dir / "pubtator_search_index"
    gnorm2_dir = cache_dir / "GNorm2"
    tmvar_dir = cache_dir / "tmVar3"

    # Initialize index manager
    index_manager = PubTatorIndexManager(str(index_dir))

    # Check indices
    indices_ok, missing = index_manager.check_indices()

    if not indices_ok:
        logger.error(f"Missing index files: {', '.join(missing)}")
        logger.error(
            f"\nIndex files not found in {index_dir}\n"
            "Please run 'pubtator-mcp-update-indices' first to download and create the index files.\n"
            "Example: pubtator-mcp-update-indices --cache-dir {cache_dir}\n"
        )
        sys.exit(1)

    # Initialize searcher
    searcher = PubTatorSearcher(index_manager)

    # Initialize GNorm2 processor
    gnorm2_processor = GNorm2Processor(gnorm2_dir)

    # Initialize tmVar processor
    tmvar_processor = TmVarProcessor(tmvar_dir)

    # Run the MCP server
    logger.info(f"Starting PubTator MCP server with cache directory: {cache_dir}")
    logger.info(f"Index directory: {index_dir}")
    logger.info(f"GNorm2 directory: {gnorm2_dir}")
    logger.info(f"tmVar3 directory: {tmvar_dir}")
    mcp.run()


if __name__ == "__main__":
    main()
