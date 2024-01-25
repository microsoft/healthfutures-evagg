"""Package defining execute.

Takes as input a yaml configuration file and runs the evidence aggregation pipeline.
"""

from .cli import run_sync

__all__ = ["run_sync"]
