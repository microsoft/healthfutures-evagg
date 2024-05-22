"""Package for interacting with literature sources."""

from .interfaces import IFindObservations, Observation, TextSection
from .observation import ObservationFinder, TruthsetObservationFinder
from .prompt_based import PromptBasedContentExtractor
from .truth_set import TruthsetContentExtractor
from .variant import HGVSVariantComparator, HGVSVariantFactory

__all__ = [
    # Content
    "PromptBasedContentExtractor",
    "TruthsetContentExtractor",
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
