from typing import Protocol, Set

from .base import Variant


class IPaperQuery(Protocol):
    def terms(self) -> Set[Variant]:
        """Get the terms in the query."""
        ...


class IPaperQueryIterator(Protocol):
    def __next__(self) -> IPaperQuery:
        """Get the next query."""
        ...

    def __iter__(self) -> "IPaperQueryIterator":
        """Get the iterator."""
        ...
