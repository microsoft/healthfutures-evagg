"""Package for interacting with reference resources."""

from .interfaces import IGeneLookupClient, IPubMedLookupClient, IVariantLookupClient
from .litvar import LitVarReference
from .ncbi import NcbiLookupClient

__all__ = [
    # Litvar.
    "LitVarReference",
    # Interfaces.
    "IGeneLookupClient",
    "IPubMedLookupClient",
    "IVariantLookupClient",
    # NCBI.
    "NcbiLookupClient",
]
