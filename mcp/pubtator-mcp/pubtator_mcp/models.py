"""Pydantic models for PubTator MCP server inputs and outputs."""

from pydantic import BaseModel, Field


class GeneNormalizationRequest(BaseModel):
    """Request to normalize a gene symbol to NCBI Gene ID."""
    gene_symbol: str = Field(description="Gene symbol to normalize (e.g., 'ACTC1', 'TP53')")


class GeneNormalizationResult(BaseModel):
    """Result of gene symbol normalization."""
    input_symbol: str = Field(description="The provided gene symbol")
    gene_id: int | None = Field(description="NCBI Gene ID if found")
    official_symbol: str | None = Field(description="The official gene symbol")
    found: bool = Field(description="Whether the gene was found")
    error: str | None = Field(default=None, description="Error message if not found")


class GeneVariantSearchRequest(BaseModel):
    """Request to search for papers with gene-disease associations and variants."""
    gene_symbol: str = Field(description="Gene symbol to search for (e.g., 'ACTC1', 'DES')")
    retmax: int | None = Field(default=None, description="Maximum number of papers to return")
    min_date: str | None = Field(default=None, description="Minimum publication date (YYYY/MM/DD format)")
    max_date: str | None = Field(default=None, description="Maximum publication date (YYYY/MM/DD format)")


class GeneVariantSearchResult(BaseModel):
    """Result of gene-variant paper search."""
    gene_symbol: str = Field(description="Official NCBI gene symbol")
    full_count: int = Field(description="Number of papers found before retmax limit")
    pmids: list[int] = Field(description="List of PubMed IDs (sorted newest first)")


class ErrorResult(BaseModel):
    """Error result for failed operations."""
    error: str = Field(description="Error message")


# Variant extraction models

class VariantExtractionRequest(BaseModel):
    """Request to extract genetic variants from biomedical text."""
    text: str = Field(description="BioC XML formatted text to extract variants from")
    gene_symbol: str | None = Field(default=None, description="Optional gene symbol to filter variants")


class VariantResult(BaseModel):
    """A single variant extraction result."""
    text: str = Field(description="Variant mention text")
    start_pos: int = Field(description="Start position in text")
    end_pos: int = Field(description="End position in text")
    ncbi_gene_id: int | None = Field(default=None, description="NCBI Gene ID if available")
    gene_symbol: str | None = Field(default=None, description="Gene symbol if available")
    hgvs: str | None = Field(default=None, description="HGVS notation if available")
    rs_id: str | None = Field(default=None, description="dbSNP RS ID if available")
