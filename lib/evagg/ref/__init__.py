"""Package for interacting with reference resources."""

from .interfaces import INcbiGeneClient, INcbiSnpClient
from .litvar import LitVarReference
from .ncbi import NcbiGeneClient, NcbiSnpClient

__all__ = [
    # Litvar.
    "LitVarReference",
    # Interfaces.
    "INcbiGeneClient",
    "INcbiSnpClient",
    # NCBI.
    "NcbiGeneClient",
    "NcbiSnpClient",
]
