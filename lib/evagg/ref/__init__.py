"""Package for interacting with reference resources."""

from .interfaces import (
    IAnnotateEntities,
    IBackTranslateVariants,
    IGeneLookupClient,
    INormalizeVariants,
    IPaperLookupClient,
    IRefSeqLookupClient,
    IValidateVariants,
    IVariantLookupClient,
)
from .litvar import LitVarReference
from .mutalyzer import MutalyzerClient
from .ncbi import NcbiLookupClient
from .refseq import NcbiReferenceLookupClient

__all__ = [
    # Litvar.
    "LitVarReference",
    # Interfaces.
    "IAnnotateEntities",
    "IGeneLookupClient",
    "IPaperLookupClient",
    "IVariantLookupClient",
    "IRefSeqLookupClient",
    "INormalizeVariants",
    "IBackTranslateVariants",
    "IValidateVariants",
    # NCBI.
    "NcbiLookupClient",
    "NcbiReferenceLookupClient",
    # Mutalyzer.
    "MutalyzerClient",
]
