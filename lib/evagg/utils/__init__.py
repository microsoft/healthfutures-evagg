"""Package for utilities."""

import os

from .logging import init_logger
from .settings import get_azure_credential, get_dotenv_settings, get_env_settings, get_settings_dict
from .web import CosmosCachingWebClient, IWebContentClient, RequestsWebContentClient

PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "content", "prompts")

__all__ = [
    "PROMPT_DIR",
    # Settings.
    "get_azure_credential",
    "get_dotenv_settings",
    "get_env_settings",
    "get_settings_dict",
    # Logging.
    "init_logger",
    # Web.
    "CosmosCachingWebClient",
    "IWebContentClient",
    "RequestsWebContentClient",
]
