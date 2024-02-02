from importlib import import_module
from types import ModuleType
from typing import Any, Dict

FACTORY_TAG = "di_factory"


class DiContainer:
    def __init__(self) -> None:
        self._modules: Dict[str, ModuleType] = {}

    def _try_import(self, module_name: str) -> ModuleType:
        if module_name not in self._modules:
            self._modules[module_name] = import_module(module_name)
        return self._modules[module_name]

    def create_instance(self, spec: Dict[str, Any], resources: Dict[str, object]) -> Any:
        """Create an instance of an object using the provided factory spec."""
        parameters: Dict[str, object] = {}
        values: Dict[str, object] = resources

        # Process initializer spec in order - resources, then parameters.
        for key, value in spec.items():
            if key in values:
                raise ValueError(f"Duplicate definition of '{key}' in spec.")

            # When we reach the factory definition, switch from
            # accumulating resources to accumulating parameters.
            if key == FACTORY_TAG:
                values = parameters
                continue

            # Look for nested factories to replace with instances.
            if isinstance(value, dict) and FACTORY_TAG in value:
                values[key] = self.create_instance(value, resources.copy())
            # Look for parameters that are resources and replace them with the resource instance.
            elif isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                resource_name = value[2:-2]
                if resource_name not in resources:
                    raise ValueError(f"Resource '{resource_name}' not defined.")
                values[key] = resources[resource_name]
            else:
                values[key] = value

        # Locate the module and instance factory.
        module_name, _, factory_name = spec[FACTORY_TAG].rpartition(".")
        module = self._try_import(module_name)
        try:
            instance_factory = getattr(module, factory_name)
        except AttributeError:
            raise TypeError(f"Module {module_name} does not define a {factory_name} entry point.")

        # Instantiate the instance with parameters.
        return instance_factory(**parameters)

    def build(self, config: Dict[str, Any]) -> Any:
        if FACTORY_TAG not in config:
            raise ValueError(f"Missing top-level '{FACTORY_TAG}' in config.")
        # Initialize the main class hierarchy.
        return self.create_instance(config, {})
