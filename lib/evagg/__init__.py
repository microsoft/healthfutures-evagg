"""The evagg core library."""

from ._base import Paper, Variant
from ._content import SimpleContentExtractor
from ._interfaces import IExtractFields, IGetPapers, IWriteOutput
from ._io import ConsoleOutputWriter, FileOutputWriter
from ._library import SimpleFileLibrary

__all__ = [
    # Interfaces.
    "IExtractFields",
    "IGetPapers",
    "IWriteOutput",
    # Base.
    "Paper",
    "Variant",
    # IO.
    "ConsoleOutputWriter",
    "FileOutputWriter",
    # Library.
    "SimpleFileLibrary",
    # Content.
    "SimpleContentExtractor",
]
