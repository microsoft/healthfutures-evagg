from typing import Sequence
from functools import cache
import os
import json

from ._interfaces import IGetPapers

class SimpleFileLibrary(IGetPapers):

    def __init__(self, collections: Sequence[str]) -> None:
        self._collections = collections

    def _load_collection(self, collection: str) -> Sequence[dict[str, str]]:
        papers = []
        # collection is a local directory, get a list of all of the json files in that directory
        for filename in os.listdir(collection):
            if filename.endswith('.json'):
                # load the json file into a dict and append it to papers
                with open(os.path.join(collection, filename), 'r') as f:
                    paper = json.load(f)
                    papers.append(paper)
        return papers
    
    @cache
    def _load(self) -> Sequence[dict[str, str]]:
        papers = []
        for collection in self._collections:
            papers.extend(self._load_collection(collection))
        print (f'Loaded {len(papers)} papers')
        print(set(papers))
        return set(papers)


    def search(self, gene: str, variant: str) -> list[dict[str, str]]:
        # Dummy implementation that returns all papers regardless of query.
        all_papers = self._load()
        return all_papers