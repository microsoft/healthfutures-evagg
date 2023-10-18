"""Base types for the evagg library."""

from ._base import Paper, Variant
from ._interfaces import IPaperQuery
from ._query import MultiQuery, Query

__all__ = [
    # Base.
    "Paper",
    "Variant",
    # Query.
    "MultiQuery",
    "Query",
    # Interfaces.
    "IPaperQuery",
]
