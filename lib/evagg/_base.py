# TODO dataclass?
# TODO should be immutable after load.
from typing import Any


class Paper:
    def __init__(self, **kwargs: Any) -> None:
        self.id = kwargs["id"]  # id is required
        self.evidence = kwargs.pop("evidence", {})
        self.citation = kwargs.get("citation")
        self.abstract = kwargs.get("abstract")
        self.props = kwargs

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Paper):
            return False
        return self.id == o.id

    def __repr__(self) -> str:
        text = self.props.get("paper_title") or self.props.get("citation") or self.props.get("abstract") or "unknown"
        return f'id: {self.id} - "{text[:15]}{"..." if len(text) > 15 else ""}"'


class Variant:
    def __init__(self, gene: str, variant: str) -> None:
        self._gene = gene
        self._variant = variant

    def __hash__(self) -> int:
        return hash(self._gene + self._variant)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Variant):
            return False
        return self._gene == o._gene and self._variant == o._variant

    def __repr__(self) -> str:
        return f"{self._gene}:{self._variant}"
