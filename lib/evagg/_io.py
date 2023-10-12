import csv
import json
import os
from typing import Sequence

from ._interfaces import IWriteOutput


class ConsoleOutputWriter(IWriteOutput):
    def __init__(self) -> None:
        pass

    def write(self, fields: dict[str, Sequence[dict[str, str]]]) -> None:
        print(json.dumps(fields, indent=4))


class FileOutputWriter(IWriteOutput):
    def __init__(self, path: str) -> None:
        self._path = path

    def write(self, fields: dict[str, Sequence[dict[str, str]]]) -> None:
        print(f"Writing output to: {self._path}")

        # TODO create parent if doesn't exist.
        parent = os.path.dirname(self._path)
        if not os.path.exists(parent):
            os.makedirs(parent)

        with open(self._path, "w") as f:
            json.dump(fields, f, indent=4)


class TableOutputWriter(IWriteOutput):
    def __init__(self, output_path: str) -> None:
        self._path = output_path

    def write(self, fields: dict[str, Sequence[dict[str, str]]]) -> None:
        print(f"Writing output to: {self._path}")

        table_lines = [variant for variant_list in fields.values() for variant in variant_list]
        if len(table_lines) == 0:
            print("No results to write")
            return

        parent = os.path.dirname(self._path)
        if not os.path.exists(parent):
            os.makedirs(parent)

        with open(self._path, "w", newline="") as tsvfile:
            writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
            writer.writerow(table_lines[0].keys())
            for line in table_lines:
                writer.writerow(line.values())
