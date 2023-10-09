from functools import cache
from pathlib import Path
from typing import Any, Dict

from lib.config import PydanticYamlModel
from lib.evagg import (
    IExtractFields,
    IGetPapers,
    IWriteOutput,
    Query,
    SimpleContentExtractor,
    SimpleFileLibrary,
    TableOutputWriter,
)


class EvAggApp:
    def __init__(self, query: Query, library: IGetPapers, extractor: IExtractFields, writer: IWriteOutput) -> None:
        self._query = query
        self._library = library
        self._extractor = extractor
        self._writer = writer

    def execute(self) -> None:
        # Get the papers that match the query.
        papers = self._library.search(self._query)

        # For all papers that match, extract the fields we want.
        fields = {paper.id: self._extractor.extract(paper) for paper in papers}

        # Write out the result.
        self._writer.write(fields)


class AppConfig(PydanticYamlModel):
    """Configuration for the app."""

    query: Dict[str, Any]
    library: Dict[str, Any]
    content: Dict[str, Any]
    output: Dict[str, Any]


class DiContainer:
    def __init__(self, config: Path) -> None:
        self._config_path = config

    @cache
    def application(self) -> EvAggApp:
        # Assemble dependencies.
        config = AppConfig.parse_yaml(self._config_path)
        query = Query(**config.query)
        library = SimpleFileLibrary(**config.library)
        extractor = SimpleContentExtractor(**config.content)
        writer = TableOutputWriter(**config.output)

        # Instantiate the app.
        return EvAggApp(query, library, extractor, writer)
