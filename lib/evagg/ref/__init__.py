"""Package for interacting with reference resources."""

from .interfaces import IGeneLookupClient, IVariantLookupClient
from .litvar import LitVarReference
from .ncbi import NcbiGeneClient, NcbiLookupClient, NcbiSnpClient

__all__ = [
    # Litvar.
    "LitVarReference",
    # Interfaces.
    "IGeneLookupClient",
    "IVariantLookupClient",
    # NCBI.
    "NcbiGeneClient",
    "NcbiLookupClient",
    "NcbiSnpClient",
]
