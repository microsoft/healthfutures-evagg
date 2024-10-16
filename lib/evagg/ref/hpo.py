import logging
import urllib.parse
from functools import cache
from typing import Dict, Sequence, Tuple

import numpy as np
from pyhpo import HPOTerm, Ontology, helper

from lib.evagg.utils import IWebContentClient

from .interfaces import ICompareHPO, IFetchHPO, ISearchHPO

logger = logging.getLogger(__name__)


class PyHPOClient(ICompareHPO, IFetchHPO):
    def __init__(self) -> None:
        # Instantiate the Ontology
        Ontology()

    @cache
    def _get_hpo_object(self, term: str) -> HPOTerm:
        return Ontology.get_hpo_object(term)

    def compare(self, subject: str, object: str, method: str = "graphic") -> float:
        term1 = self._get_hpo_object(subject)
        term2 = self._get_hpo_object(object)
        return term1.similarity_score(other=term2, method=method)

    def compare_set(
        self, subjects: Sequence[str], objects: Sequence[str], method: str = "graphic"
    ) -> Dict[str, Tuple[float, str]]:
        subject_objs = [self._get_hpo_object(s) for s in subjects]
        object_objs = [self._get_hpo_object(o) for o in objects]

        result: Dict[str, Tuple[float, str]] = {}

        for sub in subject_objs:
            comparisons = [(sub, obj) for obj in object_objs]
            batch_result = helper.batch_similarity(comparisons, kind="omim", method=method)
            if batch_result:
                argmax = np.argmax(batch_result)
                result[sub.id] = (batch_result[argmax], objects[argmax])  # type: ignore
            else:
                # No similarity scores returned, most likely because objects is empty.
                result[sub.id] = (-1, "")

        return result

    def fetch(self, query: str) -> Dict[str, str] | None:
        # Give ourselves a fighting chance to find based on name.
        if not query.startswith("HP:"):
            query = query.capitalize()

        try:
            term = self._get_hpo_object(query)
            return {"id": term.id, "name": term.name}
        except RuntimeError as e:
            logger.debug(f"Failed to retrieve HPO term {query}: {e}")
            return None

    def exists(self, query: str) -> bool:
        return bool(self.fetch(query))


class WebHPOClient(ISearchHPO):
    def __init__(self, web_client: IWebContentClient) -> None:
        self._web_client = web_client

    def _clean_query(self, query: str) -> str:
        # urllib.parse.quote doesn't get everything that upsets this service, so we'll manually fix the rest.
        # Forward slashes generate 500 errors even when encoded, replace with spaces.
        # Tildes cause 500 errors only when they're at the end of a string, even when encoded, replace with spaces.
        # Parentheses cause 500 errors even when encoded, replace with spaces.
        return urllib.parse.quote(
            query.replace("/", " ").replace("~", " ").replace("(", " ").replace(")", " ").replace(":", " ").strip()
        )

    def search(self, query: str, retmax: int = 1) -> Sequence[Dict[str, str]]:
        query = self._clean_query(query)
        url = f"https://ontology.jax.org/api/hp/search?q={query}&limit={retmax}"
        response = self._web_client.get(url, content_type="json")

        if not response["terms"]:
            return []

        return [
            {"id": term["id"], "name": term["name"], "definition": term["definition"], "synonyms": term["synonyms"]}
            for term in response["terms"]
        ]
