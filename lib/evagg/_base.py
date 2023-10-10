# TODO dataclass?
# TODO should be immutable after load.
from typing import Any


class Paper:
    def __init__(self, **kwargs: Any) -> None:
        self.id = kwargs.pop("id")  # id is required
        self.citation = kwargs.pop("citation", "")
        self.abstract = kwargs.pop("abstract", "")
        self.data = kwargs

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Paper):
            return False
        return self.id == o.id

    def __repr__(self) -> str:
        m = 10
        asuf = "..." if len(self.abstract) > m else ""

        return f'id: {self.id} - abstract: "{self.abstract[:m]}{asuf}"'


class Variant:
    def __init__(self, gene: str, variant: str) -> None:
        self._gene = gene
        self._variant = variant
