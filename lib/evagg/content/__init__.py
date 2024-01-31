"""Package for interacting with literature sources."""

from .interfaces import IFindVariantMentions
from .mention import TruthsetVariantMentionFinder, VariantMentionFinder
from .prompt_based import PromptBasedContentExtractor
from .simple import SimpleContentExtractor
from .truth_set import TruthsetContentExtractor

__all__ = [
    # Content
    "PromptBasedContentExtractor",
    "SimpleContentExtractor",
    "TruthsetContentExtractor",
    # Interfaces.
    "IFindVariantMentions",
    # Mention.
    "VariantMentionFinder",
    "TruthsetVariantMentionFinder",
]
