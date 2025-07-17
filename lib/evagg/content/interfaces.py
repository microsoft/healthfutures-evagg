from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from lib.evagg.types import HGVSVariant, Paper


class ICompareVariants(Protocol):
    def consolidate(
        self, variants: Sequence[HGVSVariant], disregard_refseq: bool = False
    ) -> dict[HGVSVariant, set[HGVSVariant]]:
        """Consolidate equivalent variants.

        Return a mapping from the retained variants to all variants collapsed into that variant.
        """
        ...  # pragma: no cover

    def compare(
        self, variant1: HGVSVariant, variant2: HGVSVariant, disregard_refseq: bool = False
    ) -> HGVSVariant | None:
        """Compare two variants to determine if they are biologically equivalent.

        If they are, return the more complete one, otherwise return None.
        """
        ...  # pragma: no cover


@dataclass(frozen=True)
class TextSection:
    section_type: str
    text_type: str
    offset: int
    text: str
    id: str


@dataclass(frozen=True)
class Observation:
    variant: HGVSVariant
    individual: str
    variant_descriptions: list[str]
    patient_descriptions: list[str]
    texts: list[TextSection]
    paper_id: str


class IFindVariants(Protocol):
    async def find_variant_descriptions(
        self,
        full_text_xml: str,
        full_text: str,
        focus_texts: Sequence[str] | None,
        gene_symbol: str,
        metadata: dict[str, Any],
    ) -> Sequence[str]:
        """Identify the genetic variants relevant to the gene_symbol described in the full text of the paper.

        Args:
            full_text_xml: The full text of the paper in BioC XML format
            full_text: The full text of the paper as plain text
            focus_texts: Optional sequence of focused text sections (e.g., tables) to prioritize
            gene_symbol: The gene symbol to find variants for
            metadata: Additional metadata about the paper and search context

        Returns:
            Sequence of variant descriptions as found in the source text.
            Downstream manipulations to make them HGVS-compliant may be required.
        """
        ...  # pragma: no cover


class IFindObservations(Protocol):
    async def find_observations(self, gene_symbol: str, paper: Paper) -> Sequence[Observation]:
        """Identify all observations relevant to `gene_sybmol` in `paper`.

        `paper` is the paper to search for relevant observations. Paper must be in the PMC-OA dataset and have
        appropriate license terms.
        """
        ...  # pragma: no cover
