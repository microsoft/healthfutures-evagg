"""The evagg core library."""

from .app import PaperQueryApp
from .content import PromptBasedContentExtractor, PromptBasedContentExtractorCached
from .interfaces import IEvAggApp, IExtractFields, IGetPapers, IWriteOutput
from .io import TableOutputWriter
from .library import PaperListLibrary, RareDiseaseFileLibrary, RareDiseaseLibraryCached
from .simple import PropertyContentExtractor, SampleContentExtractor, SimpleFileLibrary
from .truthset import TruthsetFileHandler

__all__ = [
    # Interfaces.
    "IEvAggApp",
    "IGetPapers",
    "IExtractFields",
    "IWriteOutput",
    # App.
    "PaperQueryApp",
    # IO.
    "TableOutputWriter",
    # Library.
    "SimpleFileLibrary",
    "TruthsetFileHandler",
    "RareDiseaseFileLibrary",
    "RareDiseaseLibraryCached",
    "PaperListLibrary",
    # Content.
    "PromptBasedContentExtractor",
    "PromptBasedContentExtractorCached",
    "PropertyContentExtractor",
    "SampleContentExtractor",
]
