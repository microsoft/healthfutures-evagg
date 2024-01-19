"""Base types for the evagg library."""

from .base import Paper, Variant
from .interfaces import IPaperQuery, IPaperQueryIterator
from .query import Query, QueryIterator

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
