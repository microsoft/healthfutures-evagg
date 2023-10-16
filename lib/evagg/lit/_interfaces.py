from typing import Any, Protocol, Sequence

from lib.evagg.types import IPaperQuery, Paper


class IAnnotateEntities(Protocol):
    def annotate(self, paper: Paper) -> dict[str, Any]:
        ...


# TODO, this should be keyed on Variant, not a string representation of the variant.
class IFindVariantMentions(Protocol):
    def find_mentions(self, query: IPaperQuery, paper: Paper) -> dict[str, Sequence[dict[str, Any]]]:
        """Find variant mentions relevant to query that are mentioned in `paper`.

        Returns a dictionary mapping each variant to a list of text chunks that mention it.
        """
        ...
