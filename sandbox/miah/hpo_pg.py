# %%
# pip install hpo3
from pyhpo import HPOSet, Ontology, helper

# %% Instantiate the Ontology
Ontology()  # 600 msec

# %% Compare two terms

term1 = Ontology.get_hpo_object("HP:0012469")
term2 = Ontology.get_hpo_object("HP:0007270")

print(term1.similarity_score(other=term2, method="resnik"))


# %%

termset1 = HPOSet.from_queries(["HP:0012469", "HP:0020219"])
termset2 = HPOSet.from_queries(["HP:0007270", "HP:0100763"])

print(termset1.similarity(other=termset2, method="resnik"))

# %% Let's look at the distribution of similarity scores based on comparing a given term with all of its parents and
# all of its children.

gene_sets = [g.hpo_set() for g in Ontology.genes]

all_terms_from_genes = {term for gs in gene_sets for term in gs.terms()}

comparisons = []
for term in all_terms_from_genes:
    for relative in term.parents.union(term.children):
        if term != relative:
            comparisons.append((term, relative))

method = "graphic"
batch_result = helper.batch_similarity(comparisons, kind="omim", method=method)

# %% Do the same thing, but now make an equal number of random comparisons.

import random

random_comparisons = []
for term in all_terms_from_genes:
    for _ in range(len(term.parents.union(term.children))):
        random_comparisons.append((term, random.choice(list(all_terms_from_genes))))

random_batch_result = helper.batch_similarity(random_comparisons, kind="omim", method=method)

# %% Plot.
import pandas as pd
import seaborn as sns

df = pd.DataFrame({"real": batch_result, "random": random_batch_result})
plt = sns.histplot(data=df)
plt.set_title(f"Real vs. Random similarity scores ({method}; N={len(df)})")

# %%
