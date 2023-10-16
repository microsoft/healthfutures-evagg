from typing import Sequence, Set

from lib.evagg.types import Variant

from ._interfaces import IPaperQuery


class Query(IPaperQuery):
    def __init__(self, gene: str, variant: str) -> None:
        self._variant = Variant(gene, variant)

    def terms(self) -> Set[Variant]:
        return {self._variant}


class MultiQuery(IPaperQuery):
    def __init__(self, variants: Sequence[str]) -> None:
        self._variants = {Variant(g, v) for g, v in [v.split(":") for v in variants]}

    def terms(self) -> Set[Variant]:
        return self._variants
