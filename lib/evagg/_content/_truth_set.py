from typing import Dict, Sequence, Set

from lib.evagg.types import IPaperQuery, Paper, Variant

from .._interfaces import IExtractFields


class TruthsetContentExtractor(IExtractFields):
    def __init__(self, field_map: Sequence[Dict[str, str]]) -> None:
        # Turn list of single-element key-mapping dicts into a list of tuples.
        self._field_map = [kv for [kv] in [kv.items() for kv in field_map]]

    def extract(self, paper: Paper, query: IPaperQuery) -> Sequence[Dict[str, str]]:
        def variant_match(pv: Variant, q: Set[Variant]) -> bool:
            return any(qv.gene == pv.gene if qv.variant == "*" else qv == pv for qv in q)

        extracted_fields = []
        # For each queried variant for which we have evidence in the paper...
        for v in filter(lambda v: variant_match(v, query.terms()), paper.evidence.keys()):
            # Union the evidence props with the paper props.
            properties = paper.props | paper.evidence[v]
            # Add a new evidence dict to the list, mapping evidence values to new key names.
            extracted_fields.append({out_key: properties[k] for k, out_key in self._field_map})
        return extracted_fields
