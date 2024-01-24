"""Package for interacting with reference resources."""

from .interfaces import IEntrezClient, IGeneLookupClient, IVariantLookupClient
from .litvar import LitVarReference
from .ncbi import NcbiLookupClient

__all__ = [
    # Litvar.
    "LitVarReference",
    # Interfaces.
    "IEntrezClient",
    "IGeneLookupClient",
    "IVariantLookupClient",
    # NCBI.
    "NcbiLookupClient",
]
