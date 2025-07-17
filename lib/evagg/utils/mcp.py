import logging
import re
from typing import Any

from fastmcp import Client


def _expand_env_vars(text: str, env_vars: dict[str, str]) -> str:
    """Expand environment variables in text using ${VAR} or ${VAR:-default} syntax.

    Args:
        text: Text containing environment variable references.
        env_vars: Environment variables to use for expansion.

    Returns:
        Text with environment variables expanded.
    """

    def replacer(match):
        var_name = match.group(1)
        default = match.group(2)

        # Check env_vars first, then os.environ
        if var_name in env_vars:
            return env_vars[var_name]

        return default if default is not None else match.group(0)

    # Pattern matches ${VAR} or ${VAR:-default}
    pattern = r"\$\{([^}:]+)(?::-([^}]*))?\}"
    return re.sub(pattern, replacer, str(text))


def _expand_env_vars_recursive(obj: Any, env_vars: dict[str, str]) -> Any:
    """Recursively expand environment variables in a nested data structure.

    Args:
        obj: Object to expand (dict, list, str, or other).
        env_vars: Environment variables to use for expansion.

    Returns:
        Object with environment variables expanded.
    """
    if isinstance(obj, dict):
        return {k: _expand_env_vars_recursive(v, env_vars) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars_recursive(item, env_vars) for item in obj]
    elif isinstance(obj, str):
        return _expand_env_vars(obj, env_vars)
    else:
        return obj


async def _mcp_log_handler(message) -> None:
    """Handle log messages from MCP server and route them to our logging system.

    Args:
        message: LoggingMessageNotificationParams with level, logger, and data fields
    """
    logger = logging.getLogger("mcp.server")

    # Extract fields from MCP log message (verified from mcp.types.LoggingMessageNotificationParams)
    level = (
        message.level
    )  # LoggingLevel: "debug", "info", "notice", "warning", "error", "critical", "alert", "emergency"
    data = message.data  # Any: string message or JSON serializable object
    server_logger = message.logger  # Optional logger name from server

    # Map MCP log levels to Python logging levels (from MCP protocol specification)
    level_mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "notice": logging.INFO,  # MCP-specific, map to INFO
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
        "alert": logging.CRITICAL,  # MCP-specific, map to CRITICAL
        "emergency": logging.CRITICAL,  # MCP-specific, map to CRITICAL
    }

    log_level = level_mapping.get(level, logging.INFO)

    # Format message including server logger name if available
    if server_logger:
        logger_name = f"mcp.server.{server_logger}"
        logger = logging.getLogger(logger_name)

    # Log the data (could be string or structured object)
    logger.log(log_level, "MCP Server: %s", data)


def create_mcp_client(config: dict[str, Any], env_vars: dict[str, str]) -> Client:
    """Create FastMCP client with environment variable substitution.

    Args:
        config: MCP client configuration with potential env var references.
        env_vars: Environment variables available for substitution.

    Returns:
        Configured FastMCP Client.
    """
    # Expand environment variables in the config
    if env_vars:
        config = _expand_env_vars_recursive(config, env_vars)

    # Convert snake_case to camelCase for FastMCP compatibility
    if "mcp_servers" in config:
        config["mcpServers"] = config.pop("mcp_servers")

    # Create client with custom log handler to capture server logs
    return Client(config, log_handler=_mcp_log_handler)
