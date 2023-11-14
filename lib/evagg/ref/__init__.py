"""Package for interacting with reference resources."""

from ._interfaces import INcbiGeneClient, INcbiSnpClient
from ._litvar import LitVarReference
from ._ncbi import NcbiGeneClient, NcbiSnpClient

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
