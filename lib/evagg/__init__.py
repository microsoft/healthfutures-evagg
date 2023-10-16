"""The evagg core library."""

from ._app import SynchronousLocalApp
from ._content import SemanticKernelContentExtractor, SimpleContentExtractor, TruthsetContentExtractor
from ._interfaces import IEvAggApp, IExtractFields, IGetPapers, IPaperQuery, IWriteOutput
from ._io import ConsoleOutputWriter, FileOutputWriter, TableOutputWriter
from ._library import SimpleFileLibrary, TruthsetFileLibrary

__all__ = [
    # Interfaces.
    "IEvAggApp",
    "IPaperQuery",
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
    # Content.
    "SemanticKernelContentExtractor",
    "SimpleContentExtractor",
    "TruthsetContentExtractor",
]
