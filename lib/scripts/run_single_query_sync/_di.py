from functools import cache
from pathlib import Path
from typing import Sequence, Any
import json

from ._app import EvAggApp
from lib.evagg import (
    IGetPapers, 
    IExtractFields, 
    IWriteOutput,
    ConsoleOutputWriter, 
    FileOutputWriter,
    SimpleFileLibrary,
    SimpleContentExtractor
)

class DiContainer():

    def __init__(self, config: Path) -> None:
        self._config_path = config

    @cache
    def application(self) -> EvAggApp:
        config_dict = self._config_file_to_dict(self._config_path)
        query = self._query(config_dict["query"])
        library = self._library(config_dict["library"])
        extractor = self._extractor(config_dict["content"])
        writer = self._writer(config_dict["output"])

        return EvAggApp(query, library, extractor, writer)

    @cache
    def _config_file_to_dict(self, config: Path) -> dict[str, Any]:
        return json.loads(config.read_text())

    # TODO, figure out alternative signature that supports caching here and below
    def _query(self, config: dict[str, str]) -> dict[str, str]: 
        assert "gene" in config
        assert "variant" in config
        return config

    def _library(self, config: dict[str, str]) -> IGetPapers:
        assert "collections" in config
        assert isinstance(config["collections"], Sequence)

        return SimpleFileLibrary(config["collections"])

    def _extractor(self, config: dict[str, str]) -> IExtractFields:
        assert "fields" in config
        assert isinstance(config["fields"], Sequence)

        return SimpleContentExtractor(config["fields"])
        
    def _writer(self, config: dict[str, str]) -> IWriteOutput:
        # return ConsoleOutputWriter()
        return FileOutputWriter(config["output_path"])