from typing import Any, Dict, Protocol, Sequence, Tuple

from lib.evagg.types import HGVSVariant, Paper


# TODO, this should be keyed on HGVSVariant, not a string representation of the variant.
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
