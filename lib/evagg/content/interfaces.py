from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Sequence

from lib.evagg.types import HGVSVariant, Paper


# TODO, this should be keyed on HGVSVariant, not a string representation of the variant.
class IFindVariantMentions(Protocol):
    def find_mentions(self, query: str, paper: Paper) -> Dict[HGVSVariant, Sequence[Dict[str, Any]]]:
        """Find variant mentions relevant to query that are mentioned in `paper`.

        Returns a dictionary mapping each variant to a list of text chunks that mention it.
        """
        ...


@dataclass(frozen=True)
class VariantObservation:
    """A Representation of a topic in a paper pertaining to an observation of a genetic variant in an individual."""

    variant: HGVSVariant
    variant_identifiers: List[str]  # Original representations of variant in the paper.
    individual_identifier: str
    mentions: List[str]


class IFindVariantObservations(Protocol):
    def find_variant_observations(self, query: str, paper: Paper) -> Sequence[VariantObservation]:
        """Find variant observations relevant to query that are mentioned in `paper`.

        Returns a list of VariantObservations.
        """
        ...
