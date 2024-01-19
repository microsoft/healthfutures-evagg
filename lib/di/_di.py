from importlib import import_module
from types import ModuleType
from typing import Any, Dict


class DiContainer:
    def __init__(self) -> None:
        self._modules: Dict[str, ModuleType] = {}

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

    def build(self, config: Dict[str, Any]) -> Any:
        return self.create_class_instance(config)
