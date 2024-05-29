import os
from pathlib import Path
from typing import Any, Dict, Hashable, Mapping, Optional, Type, TypeVar

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel

# Can use typing.Self in Python 3.11+.
Model = TypeVar("Model", bound="SettingsModel")


class SettingsModel(BaseModel, Hashable):
    """Extension of pydantic `BaseModel` with YAML support.

    Adds `parse_yaml()` and `to_yaml()` methods for reading and writing YAML files.
    """

    @classmethod
    def parse_yaml(cls: Type[Model], file: Path | str) -> Model:
        """Return a new model instance parsed from a YAML `file`."""
        with open(file, "r") as f:
            return cls.parse_obj(yaml.safe_load(f))

    def to_yaml(self, file: Path | str, **kwargs: Any) -> None:
        """Write model representation to a YAML `file`.

        Additional keyword args are allowed that match the `dict()` method.
        """
        with open(file, "w") as f:
            yaml.safe_dump(self.dict(**kwargs), f)

    def __hash__(self) -> int:
        """Return a hash of the model."""
        return hash(self.json())


def get_settings(settings: Mapping, filter_prefix: Optional[str] = None) -> Dict[str, str]:
    def filter_match(k: str) -> bool:
        return k.startswith(filter_prefix) if filter_prefix else True

    def transform(k: str) -> str:
        return k.replace(filter_prefix, "").lower() if filter_prefix else k.lower()

    return {transform(k): v for k, v in settings.items() if filter_match(k)}


def get_dotenv_settings(dotenv_path: Optional[str] = None, filter_prefix: Optional[str] = None) -> Dict[str, str]:
    return get_settings(dotenv_values(dotenv_path), filter_prefix)


def get_env_settings(filter_prefix: Optional[str] = None) -> Dict[str, str]:
    return get_settings(os.environ, filter_prefix)