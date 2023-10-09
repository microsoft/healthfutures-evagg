"""The evagg core library."""

from ._base import Paper, Query
from ._content import SimpleContentExtractor
from ._interfaces import IExtractFields, IGetPapers, IWriteOutput
from ._io import ConsoleOutputWriter, FileOutputWriter, TableOutputWriter
from ._library import SimpleFileLibrary

__all__ = [
    # Interfaces.
    "IExtractFields",
    "IGetPapers",
    "IWriteOutput",
    # Base.
    "Paper",
    "Query",
    # IO.
    "ConsoleOutputWriter",
    "FileOutputWriter",
    "TableOutputWriter",
    # Library.
    "SimpleFileLibrary",
    # Content.
    "SimpleContentExtractor",
]
