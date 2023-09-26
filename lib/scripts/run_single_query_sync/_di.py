from functools import cache
from pathlib import Path
from typing import Sequence, Any
import json

from . import _app
from evagg import IGetPapers, IExtractFields, IWriteOutput
from evagg import ConsoleOutputWriter, FileOutputWriter
from evagg import SimpleFileLibrary
from evagg import SimpleContentExtractor

class DiContainer():

    def __init__(self, config: Path) -> None:
        self._config_path = config

    @cache
    def application(self) -> _app.EvAggApp:
        config_dict = self._config_file_to_dict(self._config_path)

        query = self._query(config_dict["query"])
        library = self._library(config_dict["library"])
        extractor = self._extractor(config_dict["content"])
        writer = self._writer(config_dict["output"])

        return _app.EvAggApp(query, library, extractor, writer)

    @cache
    def _config_file_to_dict(self, config: Path) -> dict[str, Any]:
        return json.loads(config.read_text())
    
    @cache 
    def _query(self, config: dict[str, str]) -> dict[str, str]: 
        assert "gene" in config
        assert "variant" in config
        return config

    @cache
    def _library(self, config: dict[str, str]) -> IGetPapers:
        assert "collections" in config
        assert isinstance(config["collections"], Sequence)

        return SimpleFileLibrary(config["collections"])

    @cache
    def _extractor(self, config: dict[str, str]) -> IExtractFields:
        assert "fields" in config
        assert isinstance(config["fields"], Sequence)

        return SimpleContentExtractor(config["fields"])
        


    @cache
    def _writer(self, config: dict[str, str]) -> IWriteOutput:
        # return ConsoleOutputWriter()
        return FileOutputWriter(config["output_path"])