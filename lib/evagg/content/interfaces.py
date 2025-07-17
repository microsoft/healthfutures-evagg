from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Sequence, Set

from lib.evagg.types import HGVSVariant, Paper


class ICompareVariants(Protocol):
    def consolidate(
        self, variants: Sequence[HGVSVariant], disregard_refseq: bool = False
    ) -> Dict[HGVSVariant, Set[HGVSVariant]]:
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
    variant_descriptions: List[str]
    patient_descriptions: List[str]
    texts: List[TextSection]
    paper_id: str


class IFindVariants(Protocol):
    async def find_variant_descriptions(
        self, full_text: str, focus_texts: Sequence[str] | None, gene_symbol: str, metadata: Dict[str, Any]
    ) -> Sequence[str]:
        """Identify the genetic variants relevant to the gene_symbol described in the full text of the paper.

        Returned variants will be _as described_ in the source text. Downstream manipulations to make them
        HGVS-compliant may be required.
        """
        ...  # pragma: no cover


class IFindObservations(Protocol):
    async def find_observations(self, gene_symbol: str, paper: Paper) -> Sequence[Observation]:
        """Identify all observations relevant to `gene_sybmol` in `paper`.

        `paper` is the paper to search for relevant observations. Paper must be in the PMC-OA dataset and have
        appropriate license terms.
        """
        ...  # pragma: no cover
