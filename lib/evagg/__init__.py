"""The evagg core library."""

from .app import SynchronousLocalApp
from .content import PromptBasedContentExtractor, SimpleContentExtractor, TruthsetContentExtractor
from .interfaces import IEvAggApp, IExtractFields, IGetPapers, IWriteOutput
from .io import ConsoleOutputWriter, FileOutputWriter, TableOutputWriter
from .library import RareDiseaseFileLibrary, SimpleFileLibrary, TruthsetFileLibrary

__all__ = [
    # Interfaces.
    "IEvAggApp",
    "IGetPapers",
    "IExtractFields",
    "IWriteOutput",
    # App.
    "SynchronousLocalApp",
    # IO.
    "ConsoleOutputWriter",
    "FileOutputWriter",
    "TableOutputWriter",
    # Library.
    "SimpleFileLibrary",
    "TruthsetFileLibrary",
    "RareDiseaseFileLibrary",
    # Content.
    "PromptBasedContentExtractor",
    "SimpleContentExtractor",
    "TruthsetContentExtractor",
]
