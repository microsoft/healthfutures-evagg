from typing import Sequence

from ._library import (
    Paper
)

from ._interfaces import (
    IExtractFields,
)

class SimpleContentExtractor(IExtractFields):
    def __init__(self, fields: Sequence[str]) -> None:
        self._fields = fields

    def _field_to_value(self, field: str) -> str:
        if field == "MOI":
            return "AD"
        if field == "Phenotype":
            return "Long face (HP:0000276)"
        if field == "Functional data":
            return "No"
        else:
            return "Unknown"
        
    def extract(self, paper: Paper) -> dict[str, str]:
        return {field: self._field_to_value(field) for field in self._fields}