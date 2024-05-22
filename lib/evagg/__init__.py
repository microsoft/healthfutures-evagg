"""The evagg core library."""

from .app import SynchronousLocalApp
from .content import PromptBasedContentExtractor, SimpleContentExtractor, TruthsetContentExtractor
from .interfaces import IEvAggApp, IExtractFields, IGetPapers, IWriteOutput
from .io import TableOutputWriter
from .library import RareDiseaseFileLibrary, RareDiseaseLibraryCached, TruthsetFileLibrary
from .simple import SimpleFileLibrary

__all__ = [
    # Interfaces.
    "IEvAggApp",
    "IGetPapers",
    "IExtractFields",
    "IWriteOutput",
    # App.
    "SynchronousLocalApp",
    # IO.
    "TableOutputWriter",
    # Library.
    "SimpleFileLibrary",
    "TruthsetFileLibrary",
    "RareDiseaseFileLibrary",
    "RareDiseaseLibraryCached",
    # Content.
    "PromptBasedContentExtractor",
    "SimpleContentExtractor",
    "TruthsetContentExtractor",
]
