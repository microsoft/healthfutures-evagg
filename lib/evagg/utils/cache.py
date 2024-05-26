import json
import logging
import os
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

from .run import get_previous_run, get_run_path

T = TypeVar("T")
Serializer = Optional[Callable[[T], Any]]
Deserializer = Optional[Callable[[Any], T]]
Validator = Optional[Callable[[T], bool]]
file_cache_root: Optional[str] = None

logger = logging.getLogger(__name__)


def _get_file_cache_root() -> str:
    global file_cache_root
    if file_cache_root is None:
        # Create a new cache directory for the current run.
        file_cache_root = os.path.join(get_run_path(), "results_cache")
        os.makedirs(file_cache_root, exist_ok=True)

    # If a cache directory exists from a prior run, ask the user if they want to use it.
    if (old_run := get_previous_run()) and os.path.exists(os.path.join(str(old_run.path), "results_cache")):
        # Print out the run record for the most recent run.
        print(f"Found existing cached run for {old_run.name} started on {old_run.start}.")
        if input("Use this cache? [y/n]: ").lower() == "y":
            logger.warning(f"Copying existing cached run results from {old_run.dir_name} to current run.")
            os.system(f"cp -r {os.path.join(str(old_run.path), 'results_cache')} {get_run_path()}")

    return file_cache_root


class ObjectFileCache(Generic[T]):
    def __init__(self, name: str, serializer: Serializer = None, deserializer: Deserializer = None):
        self._serializer = serializer or (lambda x: x)
        self._deserializer = deserializer or (lambda x: x)
        self._cache_dir = os.path.join(_get_file_cache_root(), name)
        os.makedirs(self._cache_dir, exist_ok=True)

    def get(self, key: str) -> Optional[T]:
        cache_path = os.path.join(self._cache_dir, f"{key}.json")
        if not os.path.exists(cache_path):
            return None
        with open(cache_path, "r") as f:
            return self._deserializer(json.load(f))

    def set(self, key: str, value: T) -> None:
        assert not os.path.exists(os.path.join(self._cache_dir, f"{key}.json"))
        cache_path = os.path.join(self._cache_dir, f"{key}.json")
        with open(cache_path, "w") as f:
            json.dump(self._serializer(value), f, indent=4)


class ObjectCache(Generic[T]):
    def __init__(self, validator: Validator) -> None:
        self._cache: Dict[int, T] = {}
        self._validator = validator

    def get(self, key: int) -> Optional[T]:
        value = self._cache.get(key)
        if value is None:
            return None
        if self._validator is not None and not self._validator(value):
            del self._cache[key]
            return None
        return value

    def set(self, key: int, value: T) -> None:
        self._cache[key] = value
