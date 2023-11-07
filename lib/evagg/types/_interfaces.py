from typing import Protocol, Set

from ._base import Variant


class IPaperQuery(Protocol):
    def terms(self) -> Set[Variant]:
        ...


class IPaperQueryIterator(Protocol):
    def __next__(self) -> IPaperQuery:
        ...

    def __iter__(self) -> "IPaperQueryIterator":
        ...
