from functools import cache
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

from lib.config import PydanticYamlModel
from lib.evagg import IEvAggApp


class AppConfig(PydanticYamlModel):
    """Configuration for the app."""

    app: Dict[str, Any]
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
            raise TypeError(f"Module does not define a {class_name} class")

        return class_obj(**spec)

    @cache
    def application(self) -> IEvAggApp:
        # Assemble dependencies.
        config = AppConfig.parse_yaml(self._config_path)

        app_args = {
            "query": self.create_class_instance(config.query),
            "library": self.create_class_instance(config.library),
            "extractor": self.create_class_instance(config.content),
            "writer": self.create_class_instance(config.output),
        }

        # Instantiate the app.
        app_config = config.app | app_args
        return self.create_class_instance(app_config)
