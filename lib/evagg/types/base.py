from dataclasses import dataclass
from typing import Any


# TODO dataclass?
class Paper:
    def __init__(self, **kwargs: Any) -> None:
        self.id = kwargs["id"]  # id is required, DOI
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
        text = self.props.get("paper_title") or self.citation or self.abstract or "unknown"
        return f'id: {self.id} - "{text[:15]}{"..." if len(text) > 15 else ""}"'


@dataclass(frozen=True)
class HGVSVariant:
    """A representation of a genetic variant."""

    hgvs_desc: str
    gene_symbol: str | None
    refseq: str | None
    refseq_predicted: bool
    valid: bool
    protein_consequence: "HGVSVariant | None"

    def __str__(self) -> str:
        """Obtain a string representation of the variant."""
        return f"{self.refseq}:{self.hgvs_desc}"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HGVSVariant):
            return False
        return (
            self.refseq == other.refseq
            and self.gene_symbol == other.gene_symbol
            and self._comparable() == other._comparable()
        )

    def __hash__(self) -> int:
        return hash((self.refseq, self.gene_symbol, self._comparable()))

    def _comparable(self) -> str:
        """Return a string representation of the variant description that is suitable for direct string comparison.

        This includes
        - dropping of prediction parentheses.
        - substitution of * for Ter in the three letter amino acid representation.

        For example: p.(Arg123Ter) -> p.Arg123*
        """
        # TODO, consider normalization via mutalyzer
        return self.hgvs_desc.replace("(", "").replace(")", "").replace("Ter", "*")
