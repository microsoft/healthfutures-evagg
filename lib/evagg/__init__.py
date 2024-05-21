"""The evagg core library."""

from .app import SynchronousLocalApp
from .content import PromptBasedContentExtractor, SimpleContentExtractor, TruthsetContentExtractor
from .interfaces import IEvAggApp, IExtractFields, IGetPapers, IWriteOutput
from .io import TableOutputWriter
from .library import RareDiseaseFileLibrary, RemoteFileLibrary, SimpleFileLibrary, TruthsetFileLibrary

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
    "RemoteFileLibrary",
    "RareDiseaseFileLibrary",
    # Content.
    "PromptBasedContentExtractor",
    "SimpleContentExtractor",
    "TruthsetContentExtractor",
]
