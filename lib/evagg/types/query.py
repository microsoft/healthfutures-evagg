from typing import Sequence, Set

from .base import Variant
from .interfaces import IPaperQuery, IPaperQueryIterator


class Query(IPaperQuery):
    def __init__(self, variant: str) -> None:
        gene, mutation = variant.split(":")
        self._variant = Variant(gene, mutation)

    def terms(self) -> Set[Variant]:
        return {self._variant}


class QueryIterator(IPaperQueryIterator):
    def __init__(self, variants: Sequence[str]) -> None:
        self._queries = [Query(v) for v in variants]

    def __iter__(self) -> "QueryIterator":
        return self

    def __next__(self) -> IPaperQuery:
        if not self._queries:
            raise StopIteration
        return self._queries.pop(0)
