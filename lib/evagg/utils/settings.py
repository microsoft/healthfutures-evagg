import os
from typing import Any, Dict, Mapping, Optional

from azure.identity import AzureCliCredential, DefaultAzureCredential, get_bearer_token_provider
from dotenv import dotenv_values


def get_settings(settings: Mapping, filter_prefix: Optional[str] = None) -> Dict[str, str]:
    def filter_match(k: str) -> bool:
        return k.startswith(filter_prefix) if filter_prefix else True

    def transform(k: str) -> str:
        return k.replace(filter_prefix, "").lower() if filter_prefix else k.lower()

    return {transform(k): v for k, v in settings.items() if filter_match(k)}


def get_dotenv_settings(
    dotenv_path: Optional[str] = None, filter_prefix: Optional[str] = None, **additional_settings: Any
) -> Dict[str, Any]:
    """Return dotenv settings matching an optional filter prefix and appending any additional settings."""
    settings = get_settings(dotenv_values(dotenv_path), filter_prefix)
    settings.update(additional_settings)
    return settings


def get_env_settings(filter_prefix: Optional[str] = None, **additional_settings: Any) -> Dict[str, str]:
    """Return OS environment settings matching an optional filter prefix and appending any additional settings."""
    settings = get_settings(os.environ, filter_prefix)
    settings.update(additional_settings)
    return settings


def get_azure_credential(cred_type: Optional[str] = "Default", bearer_scope: Optional[str] = None) -> Any:
    """Return an Azure credential of the given type with an optional bearer scope."""
    cred: Any
    if cred_type == "Default":
        cred = DefaultAzureCredential()
    elif cred_type == "AzureCli":
        cred = AzureCliCredential()
    else:
        raise ValueError(f"Unsupported credential type: {cred_type}")
    return get_bearer_token_provider(cred, bearer_scope) if bearer_scope else cred
