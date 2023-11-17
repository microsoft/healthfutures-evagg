"""The evagg core library."""

from ._app import SynchronousLocalApp, SynchronousLocalBatchApp
from ._content._semantic_kernel import SemanticKernelContentExtractor
from ._content._simple import SimpleContentExtractor
from ._content._subject_based import SubjectBasedContentExtractor
from ._content._truth_set import TruthsetContentExtractor
from ._interfaces import IEvAggApp, IExtractFields, IGetPapers, IWriteOutput
from ._io import ConsoleOutputWriter, FileOutputWriter, TableOutputWriter
from ._library import SimpleFileLibrary, TruthsetFileLibrary

__all__ = [
    # Interfaces.
    "IEvAggApp",
    "IGetPapers",
    "IExtractFields",
    "IWriteOutput",
    # App.
    "SynchronousLocalApp",
    "SynchronousLocalBatchApp",
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
    "SubjectBasedContentExtractor",
]
