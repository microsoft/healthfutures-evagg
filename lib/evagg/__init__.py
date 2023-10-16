"""The evagg core library."""

from ._content import SemanticKernelContentExtractor, SimpleContentExtractor, TruthsetContentExtractor
from ._interfaces import IExtractFields, IGetPapers, IPaperQuery, IWriteOutput
from ._io import ConsoleOutputWriter, FileOutputWriter, TableOutputWriter
from ._library import SimpleFileLibrary, TruthsetFileLibrary

__all__ = [
    # Interfaces.
    "IPaperQuery",
    "IGetPapers",
    "IExtractFields",
    "IWriteOutput",
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
