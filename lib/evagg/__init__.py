"""The evagg core library."""

from ._app import SynchronousLocalApp
from ._content._semantic_kernel import SemanticKernelContentExtractor
from ._content._simple import SimpleContentExtractor
from ._content._truth_set import TruthsetContentExtractor
from ._interfaces import IEvAggApp, IExtractFields, IGetPapers, IPaperQuery, IWriteOutput
from ._io import ConsoleOutputWriter, FileOutputWriter, TableOutputWriter
from ._library import PubMedFileLibrary, SimpleFileLibrary, TruthsetFileLibrary

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
    "PubMedFileLibrary",
    # Content.
    "SemanticKernelContentExtractor",
    "SimpleContentExtractor",
    "TruthsetContentExtractor",
]
