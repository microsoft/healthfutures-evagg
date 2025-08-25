"""This script counts the number of unique PMIDs from an EvAgg output file that were processed as abstract-only."""

# %% Imports.

import pandas as pd

from lib.di import DiContainer

# %% Constants.

FILEPATH = ".out/run_evagg_pipeline_20250322_010043/pipeline_benchmark.tsv"

# %% Step 1: Load the TSV as a pandas DataFrame

df = pd.read_csv(FILEPATH, sep="\t", comment="#")

# %% Step 2: Instantiate an NCBI Client

ncbi_client = DiContainer().create_instance({"di_factory": "lib/config/objects/ncbi_cache.yaml"}, {})

# %% Step 3: Preprocess the DataFrame

# Drop all duplicate rows based on paper_id.
df2 = df.drop_duplicates(subset=["paper_id"]).copy()

# Make a new column, pmid, removing the prefix "pmid:" from the paper_id.
df2["pmid"] = df2["paper_id"].str.replace("pmid:", "", regex=False)

# %% Step 4: Query for can_process

df2["can_access"] = df2["pmid"].apply(
    lambda pmid: ncbi_client.fetch(pmid, include_fulltext=True).props.get("can_access", False)
)

# %% Make a new column, abstract_only, in df,that is True if df2.can_access is False for that paper_id

df2["abstract_only"] = df2["can_access"].apply(lambda x: not x)

df["abstract_only"] = df.paper_id.map(df2.set_index("paper_id")["abstract_only"])

# %%

# write the dataframe back out to an _annotated.tsv file
output_file = FILEPATH.replace(".tsv", "_annotated.tsv")
df.to_csv(output_file, sep="\t", index=False)
print(f"Wrote annotated file to {output_file}")
# %%
