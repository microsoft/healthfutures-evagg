from typing import Any, Dict, Protocol, Sequence, Set, Tuple

from lib.evagg.types import HGVSVariant, Paper


class ICompareVariants(Protocol):
    def consolidate(self, variants: Sequence[HGVSVariant]) -> Dict[HGVSVariant, Set[HGVSVariant]]:
        """Consolidate equivalent variants.

        Return a mapping from the retained variants to all variants collapsed into that variant.
        """
        ...  # pragma: no cover

    def compare(self, variant1: HGVSVariant, variant2: HGVSVariant) -> HGVSVariant | None:
        """Compare two variants to determine if they are biologically equivalent.

        If they are, return the more complete one, otherwise return None.
        """
        ...  # pragma: no cover


class IFindVariantMentions(Protocol):
    def find_mentions(self, query: str, paper: Paper) -> Dict[HGVSVariant, Sequence[Dict[str, Any]]]:
        """Find variant mentions relevant to query that are mentioned in `paper`.

        Returns a dictionary mapping each variant to a list of text chunks that mention it.
        """
        ...  # pragma: no cover


class IFindObservations(Protocol):
    def find_observations(self, query: str, paper: Paper) -> Dict[Tuple[HGVSVariant, str], Sequence[str]]:
        """Identify all observations relevant to `query` in `paper`.

        `query` should be a gene_symbol. `paper` is the paper to search for relevant observations. Paper must be in the
        PMC-OA dataset.
        """
        ...  # pragma: no cover
