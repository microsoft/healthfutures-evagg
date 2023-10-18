from typing import Any, Dict, Protocol, Sequence

from lib.evagg.types import IPaperQuery, Paper


class IAnnotateEntities(Protocol):
    def annotate(self, paper: Paper) -> Dict[str, Any]:
        ...


# TODO, this should be keyed on Variant, not a string representation of the variant.
class IFindVariantMentions(Protocol):
    def find_mentions(self, query: IPaperQuery, paper: Paper) -> Dict[str, Sequence[Dict[str, Any]]]:
        """Find variant mentions relevant to query that are mentioned in `paper`.

        Returns a dictionary mapping each variant to a list of text chunks that mention it.
        """
        ...
