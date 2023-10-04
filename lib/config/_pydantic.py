"""Extended Pydantic base classes."""

from pathlib import Path
from typing import Any, Hashable, Type, TypeVar

import yaml
from pydantic import BaseModel

# Can use typing.Self in Python 3.11+.
Model = TypeVar("Model", bound="PydanticYamlModel")


class PydanticYamlModel(BaseModel, Hashable):
    """Extension of pydantic `BaseModel` with YAML support.

    Adds `parse_yaml()` and `to_yaml()` methods for reading and writing YAML files.
    """

    @classmethod
    def parse_yaml(cls: Type[Model], file: Path | str) -> Model:
        """Return a new model instance parsed from a YAML `file`."""
        with open(file, "r") as f:
            return cls.model_validate(yaml.safe_load(f))

    def to_yaml(self, file: Path | str, **kwargs: Any) -> None:
        """Write model representation to a YAML `file`.

        Additional keyword args are allowed that match the `dict()` method.
        """
        with open(file, "w") as f:
            yaml.safe_dump(self.model_dump(**kwargs), f)

    def __hash__(self) -> int:
        """Return a hash of the model."""
        return hash(self.model_dump_json())
