import csv
import logging
import os
import sys
from datetime import datetime
from typing import Mapping, Optional, Sequence

from lib.evagg.utils.logging import get_run_root

from .interfaces import IWriteOutput

logger = logging.getLogger(__name__)


class TableOutputWriter(IWriteOutput):
    def __init__(self, tsv_name: Optional[str] = None) -> None:
        self._generated = datetime.now().astimezone()
        self._path = os.path.join(get_run_root(), f"{tsv_name}.tsv") if tsv_name else None

    def write(self, output: Sequence[Mapping[str, str]]) -> None:
        logger.info(f"Writing output to: {self._path or 'stdout'}")

        if len(output) == 0:
            logger.warning("No results to write")
            return

        if self._path:
            parent = os.path.dirname(self._path)
            if not os.path.exists(parent):
                os.makedirs(parent)

        output_stream = open(self._path, "w") if self._path else sys.stdout
        writer = csv.writer(output_stream, delimiter="\t", lineterminator="\n")
        writer.writerow([f"# Generated {self._generated.strftime('%Y-%m-%d %H:%M:%S %Z')}"])

        field_names = output[0].keys()
        writer.writerow(field_names)
        for line in output:
            # For table output, all rows must have the same keys.
            assert line.keys() == field_names
            writer.writerow(line.values())

        if self._path:
            output_stream.close()
