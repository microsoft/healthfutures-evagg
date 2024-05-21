"""The evagg core library."""

from .app import SynchronousLocalApp
from .content import PromptBasedContentExtractor, SimpleContentExtractor, TruthsetContentExtractor
from .interfaces import IEvAggApp, IExtractFields, IGetPapers, IWriteOutput
from .io import TableOutputWriter
from .library import (
    RareDiseaseFileLibrary,
    RareDiseaseLibraryCached,
    RemoteFileLibrary,
    SimpleFileLibrary,
    TruthsetFileLibrary,
)

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
    "RareDiseaseLibraryCached",
    # Content.
    "PromptBasedContentExtractor",
    "SimpleContentExtractor",
    "TruthsetContentExtractor",
]
