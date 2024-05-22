"""Package for interacting with literature sources."""

from .interfaces import IFindObservations, Observation, TextSection
from .observation import ObservationFinder, TruthsetObservationFinder
from .prompt_based import PromptBasedContentExtractor
from .variant import HGVSVariantComparator, HGVSVariantFactory

__all__ = [
    # Content
    "PromptBasedContentExtractor",
    # Interfaces.
    "IFindObservations",
    "Observation",
    "TextSection",
    # Mention.
    "HGVSVariantFactory",
    "HGVSVariantComparator",
    # Observation.
    "ObservationFinder",
    "TruthsetObservationFinder",
]
