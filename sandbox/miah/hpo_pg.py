# %%
import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyhpo import Ontology
from sklearn.metrics import roc_curve

from lib.evagg.ref import HPOReference

# %% Function definitions


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
plot = sns.histplot(data=df)
plot.set_title(f"Real vs. Random similarity scores (N={len(df)})")
plot.set_ylim(0, 3000)

# %% Determine a threshold with the maximum AUROC.

# Concatenate comparisons and random_comparisons, making a truth column where comparisons is true and random_comparisons is false.
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