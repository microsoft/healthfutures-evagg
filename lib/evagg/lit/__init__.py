"""Package for interacting with literature sources."""

from ._interfaces import IAnnotateEntities, IFindVariantMentions
from ._mention import TruthsetVariantMentionFinder, VariantMentionFinder

__all__ = [
    # Interfaces.
    "IAnnotateEntities",
    "IFindVariantMentions",
    # Mention.
    "VariantMentionFinder",
    "TruthsetVariantMentionFinder",
]
