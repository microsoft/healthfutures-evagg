import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

from .logging import get_log_root

T = TypeVar("T")
Serializer = Optional[Callable[[T], Any]]
Deserializer = Optional[Callable[[Any], T]]
Validator = Optional[Callable[[T], bool]]

logger = logging.getLogger(__name__)
file_cache_root: Optional[str] = None


def _get_file_cache_root() -> str:
    output_root = get_log_root()
    global file_cache_root
    if file_cache_root is None:
        # Extract the cache dir base name from the file argument in the command-line.
        cache_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
        # Look for the most recent matching cache directory (creation time encoded in the directory name as a suffix).
        # If a cache directory already exists, ask the user if they want to use it.
        if cache_dirs := sorted([d for d in os.listdir(output_root) if d.startswith(cache_name)], reverse=True):
            file_cache_root = os.path.join(output_root, cache_dirs[0])
            if input(f"Found existing cached pipeline run: {file_cache_root}. Use this cache? [y/n]: ").lower() == "y":
                logger.warning(f"Using existing cached pipeline results: {file_cache_root}")
                return file_cache_root
    # Otherwise, create a new cache directory with the current timestamp.
    file_cache_root = os.path.join(output_root, f"{cache_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
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
            json.dump(self._serializer(value), f)


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
