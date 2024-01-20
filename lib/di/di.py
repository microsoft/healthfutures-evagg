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

    def create_class_instance(self, module_path: str, spec: Dict[str, Any], services: Dict[str, object]) -> Any:
        module_name, _, callable_name = module_path.rpartition(".")
        module = self._try_import(module_name)

        try:
            class_initializer = getattr(module, callable_name)
        except AttributeError:
            raise TypeError(f"Module {module_name} does not define a {callable_name} class")

        # Recursively create instances of any nested classes.
        for key, value in spec.items():
            if isinstance(value, dict) and "di_class" in value:
                spec[key] = self.create_class_instance(value.pop("di_class"), value, services)

        return class_initializer(**spec)

    def build(self, config: Dict[str, Any]) -> Any:
        services = {}
        # First initialize any services and consolidate in the services dictionary.
        specs = [(key, value) for key, value in config.items() if isinstance(value, dict) and "di_service" in value]
        for key, spec in specs:
            services[key] = self.create_class_instance(spec.pop("di_service"), spec, services)
            config.pop(key)

        # Then initialize the main class hierarchy.
        return self.create_class_instance(config.pop("di_class"), config, services)
