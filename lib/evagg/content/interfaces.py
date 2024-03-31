from typing import Dict, Mapping, Protocol, Sequence, Set, Tuple

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


class IFindObservations(Protocol):
    def find_observations(self, gene_symbol: str, paper: Paper) -> Mapping[Tuple[HGVSVariant, str], Sequence[str]]:
        """Identify all observations relevant to `gene_sybmol` in `paper`.

        `paper` is the paper to search for relevant observations. Paper must be in the PMC-OA dataset and have
        appropriate license terms.
        """
        ...  # pragma: no cover
