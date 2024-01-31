"""Package for interacting with reference resources."""

from .interfaces import IAnnotateEntities, IGeneLookupClient, IPaperLookupClient, IVariantLookupClient
from .litvar import LitVarReference
from .ncbi import NcbiLookupClient

__all__ = [
    # Litvar.
    "LitVarReference",
    # Interfaces.
    "IAnnotateEntities",
    "IGeneLookupClient",
    "IPaperLookupClient",
    "IVariantLookupClient",
    # NCBI.
    "NcbiLookupClient",
]
