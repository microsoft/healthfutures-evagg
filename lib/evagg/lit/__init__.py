"""Package for interacting with literature sources."""

from ._interfaces import IAnnotateEntities, IFindVariantMentions
from ._mention import VariantMentionFinder

__all__ = [
    # Interfaces.
    "IAnnotateEntities",
    "IFindVariantMentions",
    # Mention.
    "VariantMentionFinder",
]
