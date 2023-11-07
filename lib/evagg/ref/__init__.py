"""Package for interacting with reference resources."""

from ._litvar import LitVarReference
from ._ncbi import NCBIGeneReference, NCBIVariantReference

__all__ = [
    # Litvar.
    "LitVarReference",
    # NCBI.
    "NCBIGeneReference",
    "NCBIVariantReference",
]
