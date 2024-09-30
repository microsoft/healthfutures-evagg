"""This notebook is used to consolidate discrepancy questions into a single text file.

This notebook can only be run after paper_finding/manuscript_discrepancy_list.py, 
content_extraction/manuscript_obs_discrepancy_list.py, and content_extraction/manuscript_content_discrepancy_list.py 
have been run. As it requires outputs of those scripts as inputs.
"""

# %% Imports.

import os

import pandas as pd

# %% Constants.

OUTPUT_DIR = ".out/manuscript_content_extraction"

PAPER_FINDING_DISCREPANCY_FILE = ".out/manuscript_paper_finding/discrepancies.tsv"

OBSERVATION_DISCREPANCY_FILE = ".out/manuscript_content_extraction/obs_discrepancies.tsv"

CONTENT_DISCREPANCY_ROOT = ".out/manuscript_content_extraction"

# Note here we have excluded animal_model, engineered_cells, and patient_cells_tissues. We could opt to include
# them later if we think we have time from the analysts to review.
CONTENT_DISCREPANCY_FILES = [
    "phenotype_discrepancies.tsv",
    "study_type_discrepancies.tsv",
    "variant_inheritance_discrepancies.tsv",
    "variant_type_discrepancies.tsv",
    "zygosity_discrepancies.tsv",
]

# %% Load in the souce data. It should all go into a big dataframe with two columns, pmid and question.

paper_finding_discrepancies = pd.read_csv(PAPER_FINDING_DISCREPANCY_FILE, sep="\t")[["pmid", "question"]]

observation_discrepancies = pd.read_csv(OBSERVATION_DISCREPANCY_FILE, sep="\t")[["pmid", "question"]]

discrepancies = pd.concat([paper_finding_discrepancies, observation_discrepancies])

for content_discrepancy_file in CONTENT_DISCREPANCY_FILES:
    content_discrepancy = pd.read_csv(os.path.join(CONTENT_DISCREPANCY_ROOT, content_discrepancy_file), sep="\t")[
        ["pmid", "question"]
    ]
    discrepancies = pd.concat([discrepancies, content_discrepancy])

# Order by pmid
discrepancies = discrepancies.sort_values(by="pmid")

# %% Write out the consolidated discrepancies.

count = 1
with open(os.path.join(OUTPUT_DIR, "sorted_discrepancies.txt"), "w") as f:
    for _, row in discrepancies.iterrows():
        f.write(f"{count}. ")
        f.write(row["question"])
        f.write("\n\n")
        count += 1

# %% Intentionally empty.
