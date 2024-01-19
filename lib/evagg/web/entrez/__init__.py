"""Classes for interacting with Entrez APIs."""

from .client import BioEntrezClient, BioEntrezConfig, BioEntrezDotEnvConfig
from .interfaces import IEntrezClient

__all__ = [
    # Interfaces.
    "IEntrezClient",
    # Client.
    "BioEntrezClient",
    "BioEntrezDotEnvConfig",
    "BioEntrezConfig",
]
