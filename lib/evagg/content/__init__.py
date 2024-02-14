"""Package for interacting with literature sources."""

from .interfaces import IFindVariantMentions
from .mention import (
    HGVSVariantFactory,
    HGVSVariantMention,
    TruthsetVariantMentionFinder,
    VariantMentionFinder,
    VariantTopic,
)
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
    "HGVSVariantFactory",
    "HGVSVariantMention",
    "TruthsetVariantMentionFinder",
    "VariantMentionFinder",
    "VariantTopic",
]
