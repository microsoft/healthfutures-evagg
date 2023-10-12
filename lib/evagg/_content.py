from typing import Dict, Sequence

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


class TruthsetContentExtractor(IExtractFields):
    def __init__(self, field_map: Sequence[Dict[str, str]]) -> None:
        # Turn list of single-element key-mapping dicts into a list of tuples.
        self._field_map = [kv for [kv] in [kv.items() for kv in field_map]]

    def extract(self, paper: Paper, query: IPaperQuery) -> Sequence[dict[str, str]]:
        extracted_fields = []
        # For each queried variant for which we have evidence in the paper...
        for v in query.terms() & paper.evidence.keys():
            # Union the evidence props with the paper props.
            properties = paper.props | paper.evidence[v]
            # Add a new evidence dict to the list, mapping evidence values to new key names.
            extracted_fields.append({out_key: properties[k] for k, out_key in self._field_map})
        return extracted_fields
