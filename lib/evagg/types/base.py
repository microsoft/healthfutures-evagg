from dataclasses import dataclass
from typing import Any


# TODO dataclass?
class Paper:
    def __init__(self, **kwargs: Any) -> None:
        self.id = kwargs["id"]  # id is required, DOI
        self.evidence = kwargs.pop("evidence", {})
        self.citation = kwargs.get("citation")  # TODO: determine format
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


@dataclass(frozen=True)
class HGVSVariant:
    """A representation of a genetic variant."""

    hgvs_desc: str
    gene_symbol: str | None
    refseq: str | None
    refseq_predicted: bool
    valid: bool

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
            and self.desc_no_paren() == other.desc_no_paren()
        )

    def desc_no_paren(self) -> str:
        """Return a string representation of the variant description without prediction parentheses.

        For example: p.(Arg123Cys) -> p.Arg123Cys
        """
        return self.hgvs_desc.replace("(", "").replace(")", "")
