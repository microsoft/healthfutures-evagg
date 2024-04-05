from functools import cache
from typing import Dict, Sequence, Tuple

import numpy as np
from pyhpo import HPOTerm, Ontology, helper

from .interfaces import ICompareHPO, ITranslateTextToHPO


class HPOReference(ICompareHPO, ITranslateTextToHPO):
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
            argmax = np.argmax(batch_result)
            result[sub.id] = (batch_result[argmax], objects[argmax])  # type: ignore

        return result

    def translate(self, text: str) -> str:
        # TODO
        raise NotImplementedError()