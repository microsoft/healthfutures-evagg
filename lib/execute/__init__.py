"""Package defining execute.

Takes as input a yaml configuration file and runs the evidence aggregation pipeline.
"""

from ._cli import run_sync

__all__ = ["run_sync"]
