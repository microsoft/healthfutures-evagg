import csv
import json
import logging
import os
import sys
from datetime import datetime
from typing import Mapping, Optional, Sequence

from ._interfaces import IWriteOutput

logger = logging.getLogger(__name__)


class ConsoleOutputWriter(IWriteOutput):
    def __init__(self) -> None:
        pass

    def write(self, fields: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
        print(json.dumps(fields, indent=4))


class FileOutputWriter(IWriteOutput):
    def __init__(self, path: str) -> None:
        self._path = path

    def write(self, fields: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
        logger.info(f"Writing output to: {self._path}")

        parent = os.path.dirname(self._path)
        if not os.path.exists(parent):
            os.makedirs(parent)

        with open(self._path, "w") as f:
            json.dump(fields, f, indent=4)


class TableOutputWriter(IWriteOutput):
    def __init__(self, output_path: Optional[str] = None) -> None:
        self._path = output_path

    def write(self, fields: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
        logger.info(f"Writing output to: {self._path or 'stdout'}")

        table_lines = [variant for variant_list in fields.values() for variant in variant_list]
        if len(table_lines) == 0:
            logger.warning("No results to write")
            return

        if self._path:
            parent = os.path.dirname(self._path)
            if not os.path.exists(parent):
                os.makedirs(parent)

        output_stream = open(self._path, "w") if self._path else sys.stdout
        writer = csv.writer(output_stream, delimiter="\t", lineterminator="\n")
        writer.writerow([f"# Created {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}"])
        writer.writerow(table_lines[0].keys())
        for line in table_lines:
            writer.writerow(line.values())

        output_stream.close()
