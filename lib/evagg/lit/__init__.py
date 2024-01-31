"""Package for interacting with literature sources."""

from .interfaces import IFindVariantMentions
from .mention import TruthsetVariantMentionFinder, VariantMentionFinder

__all__ = [
    # Interfaces.
    "IFindVariantMentions",
    # Mention.
    "VariantMentionFinder",
    "TruthsetVariantMentionFinder",
]
