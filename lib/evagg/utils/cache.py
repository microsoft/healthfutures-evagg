import json
import logging
import os
import subprocess  # nosec B404 # allow careful use of subprocess
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

from .run import get_previous_run, get_run_path

T = TypeVar("T")
Serializer = Optional[Callable[[T], Any]]
Deserializer = Optional[Callable[[Any], T]]
Validator = Optional[Callable[[T], bool]]
file_cache_root: Optional[str] = None

logger = logging.getLogger(__name__)
CACHE_DIR = "results_cache"


class ObjectFileCache(Generic[T]):
    def __init__(
        self,
        name: str,
        serializer: Serializer = None,
        deserializer: Deserializer = None,
        use_previous_cache: Optional[bool] = None,
    ):
        sub_path = os.path.join(CACHE_DIR, name)
        self._serializer = serializer or (lambda x: x)
        self._deserializer = deserializer or (lambda x: x)
        self._cache_dir = os.path.join(get_run_path(), sub_path)
        os.makedirs(self._cache_dir, exist_ok=True)

        # If use_previous_cache is not specifically set to False, check for a previous cache to copy over.
        if use_previous_cache is not False and (run := get_previous_run(sub_path=sub_path)):
            prompt = f"\x1b[32mFound existing {run.name} cache started on {run.start}. Use this cache? [y/n]: \x1b[0m"
            if use_previous_cache is True or input(prompt).lower() == "y":
                logger.warning(f"Copying existing {name} cache from {run.dir_name} to current run.")
                subprocess.run(
                    ["cp", "-r", os.path.join(str(run.path), sub_path), os.path.join(get_run_path(), CACHE_DIR)],
                    check=True,
                )  # nosec B603, B607 # no untrusted input/partial path

    def get(self, key: str) -> Optional[T]:
        assert all(c not in key for c in "/\\:"), f"Invalid key '{key}' for file cache."
        cache_path = os.path.join(self._cache_dir, f"{key}.json")
        if not os.path.exists(cache_path):
            return None
        with open(cache_path, "r") as f:
            return self._deserializer(json.load(f))

    def set(self, key: str, value: T) -> None:
        assert all(c not in key for c in "/\\:"), f"Invalid key '{key}' for file cache."
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
