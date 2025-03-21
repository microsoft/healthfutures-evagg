# %% Imports.

import os

import pandas as pd

# %% Contants.
input_file = "/home/azureuser/healthfutures-evagg/.out/run_evagg_pipeline_20250320_180112/pipeline_benchmark.tsv"

# %% Step 1: Copy the original TSV to a new location
backup_file = input_file.replace(".tsv", "_backup.tsv")
os.system(f"cp {input_file} {backup_file}")
print(f"Backup created at: {backup_file}")

# %% Step 2: Load the TSV as a pandas DataFrame
df = pd.read_csv(input_file, sep="\t", comment="#")

# %% Step 3: Remove rows where both 'hgvs_c' and 'hgvs_p' are NaN
df = df.dropna(subset=["hgvs_c", "hgvs_p"], how="all")

# %% Step 5: Fix format issues in the 'phenotype' column
df["phenotype"] = df["phenotype"].str.replace(r"[\[\]']", "", regex=True)

# %% Rename column id to evidence_id
df.rename(columns={"id": "evidence_id"}, inplace=True)

# %% Add in the functional studies columns
# "engineered_cells", "patient_cells_tissues", "animal_model"
# Set all values to False
df["engineered_cells"] = False
df["patient_cells_tissues"] = False
df["animal_model"] = False

# "study_type"
# Set all values to "placeholder"
df["study_type"] = "placeholder"

# %% Reorder columns
col_set = set(df.columns)

known_order = [
    "evidence_id",
    "gene",
    "paper_id",
    "hgvs_c",
    "hgvs_p",
    "paper_variant",
    "transcript",
    "validation_error",
    "individual_id",
    "citation",
    "link",
    "paper_title",
    "phenotype",
    "zygosity",
    "variant_inheritance",
    "variant_type",
    "study_type",
    "engineered_cells",
    "patient_cells_tissues",
    "animal_model",
]

new_order = known_order + list((col_set - set(known_order)))

df = df[new_order]

# %% Write the resultant DataFrame back to the original file location, note that the first line of the original file
# is a comment and should be preserved
with open(input_file, "w") as f:
    # Write the first line of the original file
    with open(backup_file, "r") as backup_f:
        first_line = backup_f.readline()
        f.write(first_line)
    # Write the DataFrame to the file
    df.to_csv(f, sep="\t", index=False, header=True)

# %%
