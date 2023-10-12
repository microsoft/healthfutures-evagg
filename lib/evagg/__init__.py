"""The evagg core library."""

from ._base import Paper, Variant
from ._content import SimpleContentExtractor, TruthsetContentExtractor
from ._interfaces import IExtractFields, IGetPapers, IPaperQuery, IWriteOutput
from ._io import ConsoleOutputWriter, FileOutputWriter, TableOutputWriter
from ._library import SimpleFileLibrary, TruthsetFileLibrary
from ._query import MultiQuery, Query

__all__ = [
    # Interfaces.
    "IPaperQuery",
    "IGetPapers",
    "IExtractFields",
    "IWriteOutput",
    # Base.
    "Paper",
    "Variant",
    # Query.
    "Query",
    "MultiQuery",
    # IO.
    "ConsoleOutputWriter",
    "FileOutputWriter",
    "TableOutputWriter",
    # Library.
    "SimpleFileLibrary",
    "TruthsetFileLibrary",
    # Content.
    "SimpleContentExtractor",
    "TruthsetContentExtractor",
]
