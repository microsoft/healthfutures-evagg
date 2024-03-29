from typing import Dict, Sequence

from lib.evagg.types import Paper

from ..interfaces import IExtractFields


class SimpleContentExtractor(IExtractFields):
    def __init__(self, fields: Sequence[str]) -> None:
        self._fields = fields

    def _field_to_value(self, field: str, paper: Paper) -> str:
        if field == "gene":
            return "CHI3L1"
        if field == "paper_id":
            return paper.id
        if field == "pmid":
            return paper.props.get("pmid", "Unknown")
        if field == "pmcid":
            return paper.props.get("pmcid", "Unknown")
        if field == "hgvs_c":
            return "c.101A>G"
        if field == "hgvs_p":
            return "p.Y34C"
        if field == "individual_id":
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

    def extract(self, paper: Paper, gene_symbol: str) -> Sequence[Dict[str, str]]:
        # Dummy implementation that returns a single variant with a static set of fields.
        return [{field: self._field_to_value(field, paper) for field in self._fields}]
