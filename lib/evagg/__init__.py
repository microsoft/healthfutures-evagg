"""The evagg core library."""

from ._app import SynchronousLocalApp, SynchronousLocalBatchApp
from ._content._prompt_based import PromptBasedContentExtractor
from ._content._simple import SimpleContentExtractor
from ._content._truth_set import TruthsetContentExtractor
from ._interfaces import IEvAggApp, IExtractFields, IGetPapers, IWriteOutput
from ._io import ConsoleOutputWriter, FileOutputWriter, TableOutputWriter
from ._library import PubMedFileLibrary, SimpleFileLibrary, TruthsetFileLibrary

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
    "PubMedFileLibrary",
    # Content.
    "PromptBasedContentExtractor",
    "SimpleContentExtractor",
    "TruthsetContentExtractor",
]
