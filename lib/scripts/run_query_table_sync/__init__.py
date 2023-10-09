"""Package defining run_query_table_sync.

Takes as input a yaml configuration file and runs the evidence aggregation pipeline synchronously.
"""

from ._cli import main

__all__ = ["main"]
