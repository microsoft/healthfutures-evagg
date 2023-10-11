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


TABLE_COLUMNS = [
    "gene_symbol",  # SEQR looks for this column name
    "paper_id",
    "hgvsc",
    "hgvsp",
    "phenotype",
    "zygosity",
    "inheritance",
    "citation",
    "study_type",
    "functional_info",
    "mutation_type",
]


class TableOutputWriter(IWriteOutput):
    def __init__(self, output_path: str) -> None:
        self._path = output_path

    def write(self, fields: dict[str, Sequence[dict[str, str]]]) -> None:
        print(f"Writing output to: {self._path}")

        table_lines = []
        # Reformat the data into a table keyed by paper/variant pair.
        for paper_id, variants in fields.items():
            for variant in variants:
                table_lines.append(
                    [
                        variant["gene"],
                        paper_id,
                        "",
                        variant["variant"],
                        variant["phenotype"],
                        "",
                        variant["MOI"],
                        "",
                        "",
                        variant["functional data"],
                        "",
                    ]
                )

        parent = os.path.dirname(self._path)
        if not os.path.exists(parent):
            os.makedirs(parent)

        with open(self._path, "w", newline="") as tsvfile:
            writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
            writer.writerow(TABLE_COLUMNS)
            for record in table_lines:
                writer.writerow(record)
