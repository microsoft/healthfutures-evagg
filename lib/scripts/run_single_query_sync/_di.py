from functools import cache
from pathlib import Path
from typing import Sequence, Any
import json

from ._app import EvAggApp
from lib.evagg import (
    IGetPapers, 
    IExtractFields, 
    IWriteOutput,
    Variant,
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

        # Assemble dependencies.
        config_dict = self._config_file_to_dict(self._config_path)
        query = self._query(
            gene=config_dict["query"]["gene"], 
            modification=config_dict["query"]["modification"]
        )
        library = self._library(
            collections=tuple(config_dict["library"]["collections"])
        )
        extractor = self._extractor(
            fields=tuple(config_dict["content"]["fields"])
        )
        writer = self._writer(
            output_path=config_dict["output"]["output_path"]
        )

        # Instantiate the app.
        return EvAggApp(query, library, extractor, writer)

    # TODO - consider frozendict and passing a config dict to each of the private dependency builders below.
    # See https://stackoverflow.com/questions/6358481/using-functools-lru-cache-with-dictionary-arguments

    @cache
    def _config_file_to_dict(self, config: Path) -> dict[str, Any]:
        return json.loads(config.read_text())

    @cache
    def _query(self, gene: str, modification: str) -> Variant: 
        return Variant(gene=gene, modification=modification)

    @cache
    def _library(self, collections: Sequence[str]) -> IGetPapers:
        return SimpleFileLibrary(collections)

    @cache
    def _extractor(self, fields: Sequence[str]) -> IExtractFields:
        return SimpleContentExtractor(fields)
        
    @cache
    def _writer(self, output_path: str) -> IWriteOutput:
        return ConsoleOutputWriter()
        # return FileOutputWriter(output_path)