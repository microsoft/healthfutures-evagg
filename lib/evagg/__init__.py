"""The evagg core library."""

from ._app import SynchronousLocalApp
from ._base import Paper, Variant
from ._content import SimpleContentExtractor, TruthsetContentExtractor
from ._di import DiContainer
from ._interfaces import IEvAggApp, IExtractFields, IGetPapers, IPaperQuery, IWriteOutput
from ._io import ConsoleOutputWriter, FileOutputWriter, TableOutputWriter
from ._library import SimpleFileLibrary, TruthsetFileLibrary
from ._query import MultiQuery, Query

__all__ = [
    # Interfaces.
    "IEvAggApp",
    "IPaperQuery",
    "IGetPapers",
    "IExtractFields",
    "IWriteOutput",
    # Base.
    "Paper",
    "Variant",
    # DI.
    "DiContainer",
    # App.
    "SynchronousLocalApp",
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
