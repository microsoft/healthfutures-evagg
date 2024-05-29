"""The evagg core library."""

from .app import PaperQueryApp
from .content import PromptBasedContentExtractor
from .interfaces import IEvAggApp, IExtractFields, IGetPapers, IWriteOutput
from .io import TableOutputWriter
from .library import RareDiseaseFileLibrary, RareDiseaseLibraryCached
from .simple import SimpleContentExtractor, SimpleFileLibrary
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
    # Content.
    "PromptBasedContentExtractor",
    "SimpleContentExtractor",
]
