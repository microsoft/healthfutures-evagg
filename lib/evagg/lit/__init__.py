"""Package for interacting with literature sources."""

from .interfaces import IAnnotateEntities, IFindVariantMentions
from .mention import TruthsetVariantMentionFinder, VariantMentionFinder

__all__ = [
    # Interfaces.
    "IAnnotateEntities",
    "IFindVariantMentions",
    # Mention.
    "VariantMentionFinder",
    "TruthsetVariantMentionFinder",
]
