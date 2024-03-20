"""Package for interacting with literature sources."""

from .interfaces import IFindObservations, IFindVariantMentions
from .mention import (  # HGVSVariantMention,; TruthsetVariantMentionFinder,; VariantMentionFinder,; VariantTopic,
    HGVSVariantFactory,
)
from .observation import ObservationFinder
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
    "IFindObservations",
    # Mention.
    "HGVSVariantFactory",
    # Observation.
    "ObservationFinder",
]
