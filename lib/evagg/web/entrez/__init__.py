"""Classes for interacting with Entrez APIs."""

from ._client import BioEntrezClient
from ._interfaces import IEntrezClient

__all__ = ["IEntrezClient", "BioEntrezClient"]
