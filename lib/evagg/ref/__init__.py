"""Package for interacting with reference resources."""

from .interfaces import IGeneLookupClient, IPaperLookupClient, IVariantLookupClient
from .litvar import LitVarReference
from .ncbi import NcbiLookupClient

__all__ = [
    # Litvar.
    "LitVarReference",
    # Interfaces.
    "IGeneLookupClient",
    "IPaperLookupClient",
    "IVariantLookupClient",
    # NCBI.
    "NcbiLookupClient",
]
