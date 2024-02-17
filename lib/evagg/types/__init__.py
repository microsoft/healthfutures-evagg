"""Base types for the evagg library."""

from .base import HGVSVariant, Paper
from .interfaces import ICreateVariants

__all__ = [
    # Base.
    "Paper",
    "HGVSVariant",
    # Interfaces.
    "ICreateVariants",
]
