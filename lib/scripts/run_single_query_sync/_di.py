from functools import cache
from pathlib import Path
from typing import Literal, Sequence

from pydantic import Field

from lib.config import PydanticYamlModel
from lib.evagg import (
    ConsoleOutputWriter,
    FileOutputWriter,
    IExtractFields,
    IGetPapers,
    IWriteOutput,
    SimpleContentExtractor,
    SimpleFileLibrary,
)
from lib.evagg.types import Query

from ._app import EvAggApp


class _QueryConfig(PydanticYamlModel):
    """Configuration for the query."""

    gene: str = Field(description="The gene of interest")
    variant: str = Field(description="The variant of interest")


class _LibraryConfig(PydanticYamlModel):
    """Configuration for the library."""

    collections: Sequence[str] = Field(description="The collections to search. Only applicable for local libraries.")


class _ContentConfig(PydanticYamlModel):
    """Configuration for the content."""

    fields: Sequence[Literal["gene", "variant", "MOI", "phenotype", "functional data"]] = Field(
        description="The fields to extract from the papers."
    )


class _OutputConfig(PydanticYamlModel):
    """Configuration for the output."""

    output_path: str | None = Field(description="The path to write the output to. If not specified, write to stdout.")


class AppConfig(PydanticYamlModel):
    """Configuration for the app."""

    query: _QueryConfig
    library: _LibraryConfig
    content: _ContentConfig
    output: _OutputConfig


class DiContainer:
    def __init__(self, config: Path) -> None:
        self._config_path = config

    @cache
    def application(self) -> EvAggApp:
        # Assemble dependencies.
        config = AppConfig.parse_yaml(self._config_path)
        query = self._query(config.query)
        library = self._library(config.library)
        extractor = self._extractor(config.content)
        writer = self._writer(config.output)

        # Instantiate the app.
        return EvAggApp(query, library, extractor, writer)

    @cache
    def _query(self, config: _QueryConfig) -> Query:
        return Query(gene=config.gene, variant=config.variant)

    @cache
    def _library(self, config: _LibraryConfig) -> IGetPapers:
        return SimpleFileLibrary(config.collections)

    @cache
    def _extractor(self, config: _ContentConfig) -> IExtractFields:
        return SimpleContentExtractor(config.fields)

    @cache
    def _writer(self, config: _OutputConfig) -> IWriteOutput:
        if config.output_path is not None:
            return FileOutputWriter(config.output_path)
        return ConsoleOutputWriter()
