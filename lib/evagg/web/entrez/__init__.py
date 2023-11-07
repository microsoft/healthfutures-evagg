"""Classes for interacting with Entrez APIs."""

from ._client import BioEntrezClient, BioEntrezConfig, BioEntrezDotEnvConfig
from ._interfaces import IEntrezClient

__all__ = [
    # Interfaces.
    "IEntrezClient",
    # Client.
    "BioEntrezClient",
    "BioEntrezDotEnvConfig",
    "BioEntrezConfig",
]
