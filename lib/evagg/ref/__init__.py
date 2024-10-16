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
from .mutalyzer import MutalyzerClient
from .ncbi import NcbiLookupClient
from .refseq import RefSeqGeneLookupClient, RefSeqLookupClient

__all__ = [
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
    "RefSeqGeneLookupClient",
    "RefSeqLookupClient",
    # Mutalyzer.
    "MutalyzerClient",
    # HPO.
    "PyHPOClient",
    "WebHPOClient",
]
