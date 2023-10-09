from functools import cache
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

from lib.config import PydanticYamlModel
from lib.evagg import IExtractFields, IGetPapers, IPaperQuery, IWriteOutput


class EvAggApp:
    def __init__(
        self, query: IPaperQuery, library: IGetPapers, extractor: IExtractFields, writer: IWriteOutput
    ) -> None:
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

    def create_class_instance(self, spec: Dict[str, Any]) -> Any:
        module = import_module("lib.evagg")
        class_name = spec.pop("class")

        try:
            class_obj = getattr(module, class_name)
        except AttributeError:
            raise TypeError(f"Module does not define a {class_name}class")

        return class_obj(**spec)

    @cache
    def application(self) -> EvAggApp:
        # Assemble dependencies.
        config = AppConfig.parse_yaml(self._config_path)
        query: Query = self.create_class_instance(config.query)
        library: IGetPapers = self.create_class_instance(config.library)
        extractor: IExtractFields = self.create_class_instance(config.content)
        writer: IWriteOutput = self.create_class_instance(config.output)

        # Instantiate the app.
        return EvAggApp(query, library, extractor, writer)
