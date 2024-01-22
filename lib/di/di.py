from importlib import import_module
from types import ModuleType
from typing import Any, Dict

CLASS_TAG = "di_class"
SERVICE_TAG = "di_service"


class DiContainer:
    def __init__(self) -> None:
        self._modules: Dict[str, ModuleType] = {}

    def _try_import(self, module_name: str) -> ModuleType:
        if module_name not in self._modules:
            self._modules[module_name] = import_module(module_name)
        return self._modules[module_name]

    def create_class_instance(self, module_path: str, spec: Dict[str, Any], services: Dict[str, object]) -> Any:
        # Locate the module and class initializer.
        module_name, _, callable_name = module_path.rpartition(".")
        module = self._try_import(module_name)
        try:
            class_initializer = getattr(module, callable_name)
        except AttributeError:
            raise TypeError(f"Module {module_name} does not define a {callable_name} class.")

        # Process initializer parameters.
        for key, value in spec.items():
            # Recursively create instances of any nested classes.
            if isinstance(value, dict) and CLASS_TAG in value:
                spec[key] = self.create_class_instance(value.pop(CLASS_TAG), value, services)
            # Look for parameters that are services and replace them with the service instance.
            if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                service_name = value[2:-2]
                if service_name not in services:
                    raise TypeError(f"Service '{service_name}' not defined.")
                spec[key] = services[service_name]

        # Instantiate the class with the parameters.
        return class_initializer(**spec)

    def build(self, config: Dict[str, Any]) -> Any:
        services: Dict[str, object] = {}
        # First initialize any services and consolidate them into the services dictionary.
        specs = [(key, value) for key, value in config.items() if isinstance(value, dict) and SERVICE_TAG in value]
        for key, spec in specs:
            services[key] = self.create_class_instance(spec.pop(SERVICE_TAG), spec, services)
            config.pop(key)

        # Then initialize the main class hierarchy.
        return self.create_class_instance(config.pop(CLASS_TAG), config, services)
