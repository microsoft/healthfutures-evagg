from typing import Sequence

from ._base import Paper
from ._interfaces import IExtractFields, IPaperQuery


class SimpleContentExtractor(IExtractFields):
    def __init__(self, fields: Sequence[str]) -> None:
        self._fields = fields

    def _field_to_value(self, field: str) -> str:
        if field == "gene":
            return "CHI3L1"
        if field == "variant":
            return "p.Y34C"
        if field == "MOI":
            return "AD"
        if field == "phenotype":
            return "Long face (HP:0000276)"
        if field == "functional data":
            return "No"
        else:
            return "Unknown"

    def extract(self, paper: Paper, query: IPaperQuery) -> Sequence[dict[str, str]]:
        # Dummy implementation that returns a single variant with a static set of fields.
        return [{field: self._field_to_value(field) for field in self._fields}]


# class JsonSampleContentExtractor(IExtractFields):
