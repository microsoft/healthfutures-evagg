# %%
import random
from functools import cache
from typing import Dict, Protocol, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyhpo import HPOTerm, Ontology, helper
from sklearn.metrics import roc_curve

# %% Function definitions


class ICompareHPO(Protocol):
    def compare(self, subject: str, object: str, method: str) -> float:
        """Compare two HPO terms using the specified method.

        HPO terms should be provided as strings, e.g. "HP:0012469"
        """
        ...

    def compare_set(self, subjects: Sequence[str], objects: Sequence[str], method: str) -> Dict[str, Tuple[float, str]]:
        """Compare two sets of HPO terms using the specified method.

        HPO terms should be provided as a sequence of strings, e.g. ["HP:0012469", "HP:0007270"]
        Will return a dictionary mapping each subject term to a tuple containing the maximum similarity score from
        objects, and the term in objects corresponding to that score.
        """
        ...


class ITranslateTextToHPO(Protocol):
    def translate(self, text: str) -> str:
        """Translate a text description to an HPO term.

        Returns the HPO term as a string, e.g. "HP:0012469"
        """
        ...


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
        return Ontology.get_hpo_object(text).id


# %% Instantiate the Ontology
reference = HPOReference()

# %% Compare two terms

term1 = "HP:0012469"
term2 = "HP:0007270"

# print(term1.similarity_score(other=term2, method="resnik"))
print(reference.compare(term1, term2))

# %%

termset1 = ["HP:0012469", "HP:0020219"]
termset2 = ["HP:0007270", "HP:0100763"]

print(reference.compare_set(termset1, termset2))

# %% Let's look at the distribution of similarity scores based on comparing a given term with all of its parents and
# all of its children.

gene_sets = [g.hpo_set() for g in Ontology.genes]

all_terms_from_genes = {term for gs in gene_sets for term in gs.terms()}

comparisons = []
for term in all_terms_from_genes:
    for relative in term.parents.union(term.children):
        comparisons.append(reference.compare(term.id, relative.id))


# %% Do the same thing, but now make an equal number of random comparisons.

random_comparisons = []
for term in all_terms_from_genes:
    for _ in range(len(term.parents.union(term.children))):
        random_comparisons.append(reference.compare(term.id, random.choice(list(all_terms_from_genes)).id))


# %% Plot.

df = pd.DataFrame({"real": comparisons, "random": random_comparisons})
# %%
plt = sns.histplot(data=df)
plt.set_title(f"Real vs. Random similarity scores (N={len(df)})")
plt.set_ylim(0, 3000)

# %% Determine a threshold with the maximum AUROC.

# Concatinate comparisons and random_comparisons, making a truth column where comparisons is true and random_comparisons is false.
cat_df = pd.concat(
    [pd.DataFrame({"score": comparisons, "truth": True}), pd.DataFrame({"score": random_comparisons, "truth": False})]
)

fpr, tpr, thresholds = roc_curve(cat_df["truth"], cat_df["score"])

# Find the threshold that maximizes the AUROC.
auroc = -1
max_threshold = 0
idx = -1
for i, threshold in enumerate(thresholds):
    tpr_i = tpr[i]
    fpr_i = fpr[i]
    if tpr_i - fpr_i > auroc:
        idx = i
        auroc = tpr_i - fpr_i
        max_threshold = threshold

print(f"Max Threshold: {max_threshold}")

# %% Plot the ROC curve.
plt.plot(fpr, tpr, color="blue")
plt.plot(fpr[idx], tpr[idx], marker="o", markersize=5, color="red")


# %%
