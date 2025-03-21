"""Script to check NCBI transcript IDs and HPO IDs in EvAgg pipeline output for hallucinations.

- this script is currently configured ONLY to be run in an interactive window because we're assuming an async context.
- preprocess_v1_reasoning_tsv.py must be run before this script to clean the o1-formatted output file.

"""

# %% Imports.
import pandas as pd
from pyhpo import Ontology

from lib.di import DiContainer
from lib.evagg.llm import IPromptClient
from lib.evagg.ref import IRefSeqLookupClient

# %% Constants.

output_path = ".out/run_evagg_pipeline_20250320_180112/pipeline_benchmark.tsv"

# %% Load the data.

df = pd.read_csv(output_path, sep="\t", comment="#")

# %% Get a list of the unique HPO IDs and associated descriptions.

split_phenos = df["phenotype"].str.split(";", expand=True)

# Turn this into a single list.
all_phenos = split_phenos.melt(value_name="phenotype")["phenotype"].dropna().drop_duplicates()

# Convert this to a dataframe by splitting on " (", the first part should be name, the second should be ID.
phenos_df = all_phenos.str.split(" \\(", expand=True)
phenos_df.columns = ["name", "id"]
phenos_df["id"] = phenos_df["id"].str.replace(")", "", regex=False)
print(f"Unique phenotypes: {len(phenos_df)}")
phenos_df.dropna(subset=["id"], inplace=True)
print(f"Unique phenotypes with IDs: {len(phenos_df)}")

# %% Build some useful instances of things.

Ontology()

llm: IPromptClient = DiContainer().create_instance(spec={"di_factory": "lib/config/objects/llm.yaml"}, resources={})

refseq: IRefSeqLookupClient = DiContainer().create_instance(
    spec={"di_factory": "lib/config/objects/refseq.yaml"}, resources={}
)

# %% Define a helper function for semantic comparison.


async def ask_llm(t1: str, t2: str) -> bool:
    """Ask the LLM if two terms are semantically equivalent."""
    prompt = f"Are the following two terms semantically equivalent?\n\n{t1}\n\n{t2}\n\nAnswer with only 'yes' or 'no'."
    response = await llm.prompt(prompt, prompt_settings={"model": "gpt-4o-mini"})
    if ("yes" not in response.strip().lower()) and ("no" not in response.strip().lower()):
        print(f"Invalid response: {response}")
    return "yes" in response.strip().lower()


# %% Do the comparison for each phenotype term.
import asyncio

phenos_df["result"] = "unknown"
phenos_df["onto_name"] = "none"

for idx, row in phenos_df.iterrows():
    # Check if the ID is a valid HPO ID.

    try:
        hpo_element = Ontology.get_hpo_object(row["id"])
    except Exception as e:
        print(f"Invalid HPO ID: {row['id']}")
        print(e)
        phenos_df.at[idx, "result"] = "invalid"
        phenos_df.at[idx, "onto_name"] = "invalid"
        continue

    phenos_df.at[idx, "onto_name"] = hpo_element.name

    if asyncio.run(ask_llm(row["name"], hpo_element.name)):
        phenos_df.at[idx, "result"] = "match"
    else:
        phenos_df.at[idx, "result"] = "mismatch"

print(phenos_df.result.value_counts())

# %% now let's look at transcript ids


df["real_transcript"] = df["gene"].apply(lambda x: refseq.transcript_accession_for_symbol(x))

df["transcript_match"] = df.apply(lambda row: row.transcript.split(".")[0] == row.real_transcript.split(".")[0], axis=1)

print(df.transcript_match.value_counts())

# %% Intentionally blank.
