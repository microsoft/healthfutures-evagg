from dataclasses import dataclass
from typing import Any, List


# TODO dataclass?
class Paper:
    def __init__(self, **kwargs: Any) -> None:
        self.id = kwargs["id"]  # id is required
        self.props = kwargs

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Paper):
            return False
        return self.id == o.id

    def __repr__(self) -> str:
        text = self.props.get("title") or self.props.get("citation") or self.props.get("abstract") or "unknown"
        return f'id: {self.id} - "{text[:15]}{"..." if len(text) > 15 else ""}"'


@dataclass(frozen=True)
class HGVSVariant:
    """A representation of a genetic variant."""

    hgvs_desc: str
    gene_symbol: str | None
    refseq: str | None
    refseq_predicted: bool
    valid: bool
    validation_error: str | None
    # TODO, consider subclasses for different variant types.
    protein_consequence: "HGVSVariant | None"
    coding_equivalents: "List[HGVSVariant]"

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

    def get_unique_id(self, prefix: str, suffix: str) -> str:
        # Build a unique id and make it URL-safe.
        id = f"{prefix}_{self.hgvs_desc}_{suffix}".replace(" ", "")
        return id.replace(":", "-").replace("/", "-").replace(">", "-")
