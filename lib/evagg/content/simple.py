from typing import Dict, Sequence

from lib.evagg.types import Paper

from ..interfaces import IExtractFields


class SimpleContentExtractor(IExtractFields):
    def __init__(self, fields: Sequence[str]) -> None:
        self._fields = fields

    def _field_to_value(self, field: str) -> str:
        if field == "gene":
            return "CHI3L1"
        if field == "paper_id":
            return "10.1016/j.dib.2019.104311"
        if field == "hgvs_c":
            return "c.101A>G"
        if field == "hgvs_p":
            return "p.Y34C"
        if field == "subject_id":
            return "unknown"
        if field == "phenotype":
            return "Long face (HP:0000276)"
        if field == "zygosity":
            return "Heterozygous"
        if field == "variant_inheritance":
            return "AD"
        if field == "functional data":
            return "No"
        else:
            return "Unknown"

    def extract(self, paper: Paper, query: str) -> Sequence[Dict[str, str]]:
        # Dummy implementation that returns a single variant with a static set of fields.
        return [{field: self._field_to_value(field) for field in self._fields}]
