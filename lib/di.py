from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

import yaml

FACTORY_TAG = "di_factory"


class DiContainer:
    def __init__(self) -> None:
        self._modules: Dict[str, ModuleType] = {}

    def _try_import(self, module_name: str) -> ModuleType:
        if module_name not in self._modules:
            self._modules[module_name] = import_module(module_name)
        return self._modules[module_name]

    def _nested_update(self, d: Dict, u: Dict) -> Dict:
        """Recursively update a nested dictionary."""
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._nested_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def _instance_from_yaml(self, factory: str, parameters: Dict[str, object], resources: Dict[str, object]) -> Any:
        """Create an instance from a yaml file sub-spec."""
        if not Path(factory).exists():
            # Check if prefixing path with lib/config exists.
            if (Path("lib/config") / Path(factory)).exists():
                factory = str(Path("lib/config") / Path(factory))
            else:
                raise FileNotFoundError(f"Factory config file {factory} not found.")

        # Read in the spec dictionary from yaml.
        with open(Path(factory), "r") as f:
            yaml_object = yaml.safe_load(f)

        # If it's not a dictionary, treat it as a simple object.
        if not isinstance(yaml_object, dict):
            return yaml_object

        # Parameters to the yaml are considered overrides to items inside the yaml.
        yaml_object = self._nested_update(yaml_object, parameters)
        instance = self.create_instance(yaml_object, resources.copy())
        return instance

    def _instance_from_module(self, factory: str, parameters: Dict[str, object]) -> Any:
        """Create an instance from a module and factory entrypoint."""
        # Locate the module and instance factory from module path.
        module_name, _, factory_name = factory.rpartition(".")
        module = self._try_import(module_name)
        try:
            instance_factory = getattr(module, factory_name)
        except AttributeError:
            raise TypeError(f"Module {module_name} does not define a {factory_name} entry point.")
        # Instantiate the instance with parameters.
        instance = instance_factory(**parameters)
        return instance

    def create_instance(self, spec: Dict[str, Any], resources: Dict[str, object]) -> Any:
        """Create an instance of an object using the provided factory spec."""
        parameters: Dict[str, object] = {}
        values: Dict[str, object] = resources

        if FACTORY_TAG not in spec:
            raise ValueError(f"Missing '{FACTORY_TAG}' in instance spec.")
        factory: str = spec[FACTORY_TAG]

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

        if factory.endswith(".yaml"):
            instance = self._instance_from_yaml(factory, parameters, resources)
        else:
            instance = self._instance_from_module(factory, parameters)

        return instance
