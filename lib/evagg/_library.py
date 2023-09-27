from typing import Sequence
from functools import cache
import os
import json

from ._interfaces import IGetPapers

# TODO dataclass?
# TODO should be immutable after load.
class Paper():
    def __init__(self, id: str, citation: str, abstract: str) -> None:
        self.id = id
        self.citation = citation
        self.abstract = abstract

    def __repr__(self) -> str:
        m = 10
        csuf = "..." if len(self.citation) > m else ""
        asuf = "..." if len(self.abstract) > m else ""

        return f"id: {self.id} - abstract: \"{self.abstract[:m]}{asuf}\""
    
    # def __eq__(self, b: 'Paper') -> bool:
    #     return self.id == b.id
    
    @classmethod
    def from_dict(cls, values: dict[str, str]) -> 'Paper':
        return Paper(id=values['id'], citation=values['citation'], abstract=values['abstract'])


class SimpleFileLibrary(IGetPapers):

    def __init__(self, collections: Sequence[str]) -> None:
        self._collections = collections

    def _load_collection(self, collection: str) -> dict[str, Paper]:
        papers = {}
        # collection is a local directory, get a list of all of the json files in that directory
        for filename in os.listdir(collection):
            if filename.endswith('.json'):
                # load the json file into a dict and append it to papers
                with open(os.path.join(collection, filename), 'r') as f:
                    paper = Paper.from_dict(json.load(f))
                    papers[paper.id] = paper
        return papers
    
    @cache
    def _load(self) -> dict[str, Paper]:
        papers = {}
        for collection in self._collections:
            papers.update(self._load_collection(collection))
        
        return papers

    def search(self, gene: str, variant: str) -> Sequence[Paper]:
        # Dummy implementation that returns all papers regardless of query.
        all_papers = self._load().values()
        return all_papers