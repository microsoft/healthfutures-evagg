"""The evagg core library."""

from ._content.prompt_based import PromptBasedContentExtractor
from ._content.simple import SimpleContentExtractor
from ._content.truth_set import TruthsetContentExtractor
from .app import SynchronousLocalApp
from .interfaces import IEvAggApp, IExtractFields, IGetPapers, IWriteOutput
from .io import ConsoleOutputWriter, FileOutputWriter, TableOutputWriter
from .library import RemoteFileLibrary, SimpleFileLibrary, TruthsetFileLibrary

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
    "RemoteFileLibrary",
    # Content.
    "PromptBasedContentExtractor",
    "SimpleContentExtractor",
    "TruthsetContentExtractor",
]
