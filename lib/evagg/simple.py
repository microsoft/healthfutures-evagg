import json
import os
from typing import Any, Dict, Sequence

from lib.evagg.types import Paper

from .interfaces import IExtractFields, IGetPapers


class SimpleFileLibrary(IGetPapers):
    def __init__(self, collections: Sequence[str]) -> None:
        self._collections = collections

    def _load_collection(self, collection: str) -> Dict[str, Paper]:
        papers = {}
        # collection is a local directory, get a list of all of the json files in that directory
        for filename in os.listdir(collection):
            if filename.endswith(".json"):
                # load the json file into a dict and append it to papers
                with open(os.path.join(collection, filename), "r") as f:
                    paper = Paper(**json.load(f))
                    papers[paper.id] = paper
        return papers

    def _load(self) -> Dict[str, Paper]:
        papers = {}
        for collection in self._collections:
            papers.update(self._load_collection(collection))

        return papers

    def get_papers(self, query: Dict[str, Any]) -> Sequence[Paper]:
        # Dummy implementation that returns all papers regardless of query.
        all_papers = list(self._load().values())
        return all_papers


class SimpleContentExtractor(IExtractFields):
    def __init__(self, fields: Sequence[str]) -> None:
        self._fields = fields

    def _field_to_value(self, field: str, paper: Paper, gene_sybmol: str) -> str:
        if field == "gene":
            return gene_sybmol
        if field == "paper_id":
            return paper.id
        if field == "paper_disease_category":
            return paper.props.get("disease_category", "Unknown")
        if field == "paper_disease_categorizations":
            return json.dumps(paper.props.get("disease_categorizations", {}))
        if field == "pmid":
            return paper.props.get("pmid", "")
        if field == "pmcid":
            return paper.props.get("pmcid", "")
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
        return [{field: self._field_to_value(field, paper, gene_symbol) for field in self._fields}]
