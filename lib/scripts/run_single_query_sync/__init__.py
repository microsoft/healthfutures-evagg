"""Package defining run_single_query_sync.

Takes as input a json configuration file and runs the evidence aggregation pipeline synchronously.
"""

from ._cli import main

__all__ = ["main"]
