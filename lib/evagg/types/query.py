from typing import List, Sequence, Set, Union

from .base import Variant
from .interfaces import IPaperQuery, IPaperQueryIterator


class Query(IPaperQuery):
    def __init__(self, variants: Union[str, List[str]]) -> None:
        if isinstance(variants, str):
            variants = [variants]
        self._variants = {Variant(g, m) for variant in variants for g, m in [variant.split(":")]}

    def terms(self) -> Set[Variant]:
        return self._variants


class QueryIterator(IPaperQueryIterator):
    def __init__(self, variants: Sequence[str]) -> None:
        self._queries = [Query(v) for v in variants]

    def __iter__(self) -> "QueryIterator":
        return self

    def __next__(self) -> IPaperQuery:
        if not self._queries:
            raise StopIteration
        return self._queries.pop(0)
