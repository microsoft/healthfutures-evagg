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

    return Client(config)
