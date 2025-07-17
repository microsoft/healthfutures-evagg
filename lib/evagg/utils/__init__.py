"""Package for utilities."""

from .logging import init_logger
from .settings import get_azure_credential, get_dotenv_settings, get_env_settings
from .web import CosmosCachingWebClient, IWebContentClient, RequestsWebContentClient
from .mcp import create_mcp_client

__all__ = [
    # Settings.
    "get_azure_credential",
    "get_dotenv_settings",
    "get_env_settings",
    # Logging.
    "init_logger",
    # MCP.
    "create_mcp_client",
    # Web.
    "CosmosCachingWebClient",
    "IWebContentClient",
    "RequestsWebContentClient",
]
