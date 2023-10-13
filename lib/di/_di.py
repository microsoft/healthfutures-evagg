from functools import cache
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

import yaml


class DiContainer:
    def __init__(self, config: Path) -> None:
        self._module = import_module("lib.evagg")
        self._config_path = config

    def create_class_instance(self, spec: Dict[str, Any]) -> Any:
        class_name = spec.pop("di_class")

        try:
            class_obj = getattr(self._module, class_name)
        except AttributeError:
            raise TypeError(f"Module does not define a {class_name} class")

        # Recursively create instances of any nested classes.
        for key, value in spec.items():
            if isinstance(value, dict) and "di_class" in value:
                spec[key] = self.create_class_instance(value)

        return class_obj(**spec)

    @cache
    def build(self) -> Any:
        # Read in the config dictionary.
        with open(self._config_path, "r") as f:
            config = yaml.safe_load(f)
        # Instantiate the app.
        return self.create_class_instance(config)
