"""Package for interacting with reference resources."""

from .hpo import PyHPOClient, WebHPOClient
from .interfaces import (
    IAnnotateEntities,
    IBackTranslateVariants,
    ICompareHPO,
    IFetchHPO,
    IGeneLookupClient,
    INormalizeVariants,
    IPaperLookupClient,
    IRefSeqLookupClient,
    ISearchHPO,
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
    "IFetchHPO",
    "ISearchHPO",
    # NCBI.
    "NcbiLookupClient",
    "NcbiReferenceLookupClient",
    # Mutalyzer.
    "MutalyzerClient",
    # HPO.
    "PyHPOClient",
    "WebHPOClient",
]
