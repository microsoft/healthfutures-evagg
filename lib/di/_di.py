from functools import cache
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional

import yaml


class DiContainer:
    def __init__(self, config: Path, overrides: Optional[Dict[str, Any]]) -> None:
        self._modules: Dict[str, ModuleType] = {}
        self._config_path = config
        self._overrides = overrides or {}

    def _try_import(self, module_name: str) -> ModuleType:
        if module_name not in self._modules:
            self._modules[module_name] = import_module(module_name)
        return self._modules[module_name]

    def create_class_instance(self, spec: Dict[str, Any]) -> Any:
        module_name, _, class_name = spec.pop("di_class").rpartition(".")

        module = self._try_import(module_name)

        try:
            class_obj = getattr(module, class_name)
        except AttributeError:
            raise TypeError(f"Module {module_name} does not define a {class_name} class")

        # Recursively create instances of any nested classes.
        for key, value in spec.items():
            if isinstance(value, dict) and "di_class" in value:
                spec[key] = self.create_class_instance(value)

        return class_obj(**spec)

    def _nested_update(self, d: Dict, u: Dict) -> Dict:
        """Recursively update a nested dictionary."""
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._nested_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    @cache
    def build(self) -> Any:
        # Read in the config dictionary.
        with open(self._config_path, "r") as f:
            config = yaml.safe_load(f)

        # Merge in any overrides.
        config = self._nested_update(config, self._overrides)

        # Instantiate the app.
        return self.create_class_instance(config)
