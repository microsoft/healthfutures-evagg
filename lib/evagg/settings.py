import os
from typing import Dict, Mapping, Optional

from dotenv import dotenv_values


def get_settings(settings: Mapping, filter_prefix: Optional[str] = None) -> Dict[str, Optional[str]]:
    def filter_match(k: str) -> bool:
        return k.startswith(filter_prefix) if filter_prefix else True

    def transform(k: str) -> str:
        return k.replace(filter_prefix, "").lower() if filter_prefix else k.lower()

    return {transform(k): v for k, v in settings.items() if filter_match(k)}


def get_dotenv_settings(
    dotenv_path: Optional[str] = None, filter_prefix: Optional[str] = None
) -> Dict[str, Optional[str]]:
    return get_settings(dotenv_values(dotenv_path), filter_prefix)


def get_env_settings(filter_prefix: Optional[str] = None) -> Dict[str, Optional[str]]:
    return get_settings(os.environ, filter_prefix)
