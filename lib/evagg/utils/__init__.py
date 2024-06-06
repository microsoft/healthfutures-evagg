"""Package for interacting with services."""

from .logging import init_logger
from .settings import get_azure_credential, get_dotenv_settings, get_env_settings, get_settings_dict
from .web import CosmosCachingWebClient, IWebContentClient, RequestsWebContentClient

__all__ = [
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
