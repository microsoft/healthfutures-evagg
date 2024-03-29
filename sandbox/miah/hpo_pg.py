# # %% Download latest phenio.db from HPO site.
# # !wget https://data.monarchinitiative.org/monarch-kg/latest/phenio.db.gz -O .ref/phenio.db.gz
# # !gunzip .ref/phenio.db.gz

# # %% Imports.
# # from semsimian import Semsimian
# from semsimian.semsimian import Semsimian
# # %% Initialize things.

# db = ".ref/phenio.db"

# #  Predicates basicly define what types of edges in the graph should be considered a "link".
# # For HPO we are only interested in the "rdfs:subClassOf" predicate.
# predicates = ["rdfs:subClassOf"]

# semsimian = Semsimian(
#     spo=None,
#     predicates=predicates,
#     resource_path=db,
# )
# # %% Basic test

# # some example HPO sets
# hpo1 = {"HP:0012469", "HP:0020219"}
# hpo2 = {"HP:0007270", "HP:0100763"}

# # Run a termset comparison. First run sets up some internal data structures, so it's slower.
# full_result = semsimian.termset_pairwise_similarity(
#     set(hpo1),
#     set(hpo2),
# )

# print(full_result)

# # %%

# semsimian.

# ########

# %%
# pip install hpo3
from pyhpo import HPOSet, HPOTerm, Ontology, helper

# %% Instantiate the Ontology
Ontology()

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

# subset to the first 100 elements.
all_terms_from_genes = set(list(all_terms_from_genes)[:1000])

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

# %% Plot these two distributions using seaborn

# %% generate a pandas dataframe with 2 columns, named real and random and 30,000 rows with a simple numeric index.
import pandas as pd
import seaborn as sns

# import matplotlib.pyplot as plt

# sns.histplot(batch_result, color="blue", label="Real comparisons")
# sns.histplot(random_batch_result, color="red", label="Random comparisons")
# plt.show()


df = pd.DataFrame({"real": batch_result, "random": random_batch_result})

# Plot hte dataframe using seaborn
sns.histplot(data=df)

# %%
