"""Package for interacting with reference resources."""

from .hpo import HPOReference
from .interfaces import (
    IAnnotateEntities,
    IBackTranslateVariants,
    ICompareHPO,
    IGeneLookupClient,
    INormalizeVariants,
    IPaperLookupClient,
    IRefSeqLookupClient,
    ITranslateTextToHPO,
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
    "ICompareHPO",
    "ITranslateTextToHPO",
    # NCBI.
    "NcbiLookupClient",
    "NcbiReferenceLookupClient",
    # Mutalyzer.
    "MutalyzerClient",
    # HPO.
    "HPOReference",
]
