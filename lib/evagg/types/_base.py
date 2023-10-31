# TODO dataclass?
# TODO should be immutable after load.
from typing import Any


class Paper:
    def __init__(self, **kwargs: Any) -> None:
        self.id = kwargs["id"]  # id is required, DOI
        self.evidence = kwargs.pop("evidence", {})
        self.citation = kwargs.get("citation") # determine format
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
        self.gene = gene
        self.variant = variant

    def __hash__(self) -> int:
        return hash(self.gene + self.variant)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Variant):
            return False
        return self.gene == o.gene and self.variant == o.variant

    def __repr__(self) -> str:
        return f"{self.gene}:{self.variant}"
