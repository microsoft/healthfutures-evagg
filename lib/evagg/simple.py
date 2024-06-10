import json
import logging
import os
from functools import cache
from typing import Any, Dict, List, Sequence

from lib.evagg.types import Paper

from .interfaces import IExtractFields, IGetPapers

logger = logging.getLogger(__name__)


class PropertyContentExtractor(IExtractFields):
    PAPER_TO_EVIDENCE_KEYS = {"id": "paper_id", "title": "paper_title"}

    def __init__(self, fields: Sequence[str]) -> None:
        self._fields = fields

    @property
    def fields(self) -> Sequence[str]:
        return self._fields

    def get_evidence(self, paper: Paper, gene_symbol: str) -> Sequence[Dict[str, str]]:
        # Default implementation just returns a single set with the gene and the mapped paper properties.
        return [{"gene": gene_symbol, **{self.PAPER_TO_EVIDENCE_KEYS.get(k, k): v for k, v in paper.props.items()}}]

    def extract(self, paper: Paper, gene_symbol: str) -> Sequence[Dict[str, str]]:
        evidence = self.get_evidence(paper, gene_symbol)
        if missing_fields := set(self.fields) - set(evidence[0].keys()):
            raise ValueError(f"Unsupported extraction fields: {missing_fields}")
        logger.debug(f"Extracting fields from paper {paper.id} for gene {gene_symbol}: {len(evidence)} instances.")
        return [{f: ev[f] for f in self.fields} for ev in evidence]


class SimpleFileLibrary(IGetPapers):
    def __init__(self, collections: Sequence[str]) -> None:
        self._collections = collections

    @cache
    def _load_collections(self) -> List[Paper]:
        papers = []
        # Read in each json file in each collection as a Paper object.
        for file in [os.path.join(c, f) for c in self._collections for f in os.listdir(c) if f.endswith(".json")]:
            papers.append(Paper(**json.load(open(file, "r"))))
        return papers

    def get_papers(self, query: Dict[str, Any]) -> Sequence[Paper]:
        logger.debug(f"Getting papers for query: {query}")
        # Dummy implementation that returns all papers regardless of query.
        return self._load_collections()


class SampleContentExtractor(PropertyContentExtractor):
    def get_evidence(self, paper: Paper, gene_symbol: str) -> Sequence[Dict[str, str]]:
        props = super().get_evidence(paper, gene_symbol)
        # Add in some random variant properties as sample data.
        props[0]["hgvs_c"] = "c.101A>G"
        props[0]["zygosity"] = "Heterozygous"
        props[0]["variant_inheritance"] = "AD"
        props[0]["phenotype"] = "Long face (HP:0000276)"
        props[0]["individual_id"] = str(hash(f"{paper.id}{gene_symbol}") % 10000)
        return props
