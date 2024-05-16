from typing import Dict, Sequence

from lib.evagg.types import Paper

from ..interfaces import IExtractFields


class TruthsetContentExtractor(IExtractFields):
    def __init__(self, field_map: Sequence[Dict[str, str]]) -> None:
        # Turn list of single-element key-mapping dicts into a list of tuples.
        self._field_map = [kv for [kv] in [kv.items() for kv in field_map]]

    def extract(self, paper: Paper, gene_symbol: str) -> Sequence[Dict[str, str]]:
        """Extract properties from the evidence bags populated on the truthset Paper object."""

        def _get_props(evidence: Dict[str, str]) -> Dict[str, str]:
            """Extract the requested evidence properties from the paper and truthset evidence bag."""
            return {out_key: (paper.props | evidence)[k] for k, out_key in self._field_map}

        # For each evidence set in the paper that has a matching gene, extract the evidence properties.
        extracted_fields = [_get_props(ev) for ev in paper.evidence.values() if ev["gene"] == gene_symbol]
        return extracted_fields
