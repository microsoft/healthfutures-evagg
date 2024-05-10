"""Package for interacting with literature sources."""

from .interfaces import IFindObservations, Observation
from .observation import ObservationFinder, TruthsetObservationFinder
from .prompt_based import PromptBasedContentExtractor
from .simple import SimpleContentExtractor
from .truth_set import TruthsetContentExtractor
from .variant import HGVSVariantComparator, HGVSVariantFactory

__all__ = [
    # Content
    "PromptBasedContentExtractor",
    "SimpleContentExtractor",
    "TruthsetContentExtractor",
    # Interfaces.
    "IFindObservations",
    "Observation",
    # Mention.
    "HGVSVariantFactory",
    "HGVSVariantComparator",
    # Observation.
    "ObservationFinder",
    "TruthsetObservationFinder",
]
