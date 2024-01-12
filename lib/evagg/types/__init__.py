"""Base types for the evagg library."""

from ._base import Paper, Variant
from ._interfaces import IPaperQuery, IPaperQueryIterator
from ._query import Query, QueryIterator

__all__ = [
    # Base.
    "Paper",
    "Variant",
    # Query.
    "Query",
    "QueryIterator",
    # Interfaces.
    "IPaperQuery",
    "IPaperQueryIterator",
]
