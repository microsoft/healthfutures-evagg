"""Update the V1 truthset based on the error analysis results.

The primary output of the error analysis process was a set of manually curated "facts" that it turns out
were not actually correct in the v1 truthset. For example a a paper should have been included but it wasn't,
or a variant was incorrectly associated with the query gene and should actually be removed.

The purpose of this script is to take in the output of the error analysis process and generate a checklist necessary
to revise the truthset.

The inputs to this script are:
- The complete list of discrepancies identified during benchmarking
- The individual analyst responses from the error analysis process
- The consensus resolution from the error analysis process
"""

# %% Imports.

import pandas as pd

from lib.di import DiContainer
from lib.evagg.ref import IPaperLookupClient
from scripts.truthset.error_analysis.utils import parse_error_analysis_excel

# %% Constants.

# Generated during the benchmarking process, this is a list of all errors identified comparing repeated
# pipeline runs to the v1 truthset.
PIPELINE_ERROR_LIST = "data/error_analysis/all_sorted_discrepancies.tsv"

# The individual analyst responses from the error analysis process.
ANALYST_RESPONSES_FILES = [
    "data/error_analysis/ana1_error_analysis_worksheet.csv",
    "data/error_analysis/ana2_error_analysis_worksheet.csv",
    "data/error_analysis/ana3_error_analysis_worksheet.csv",
]

# The consensus resolution file from the error analysis process. This contains final resolution for any inter-analyst
# disagreements during error analysis.
CONSENSUS_RESOLUTION_FILE = "data/error_analysis/resolved_discrepancies.tsv"

# The original v1 truthset. This will serve as a starting point for the updated truthset.
TRUTHSET_ROOT = "data/v1"

# %% Functions and shared objects.

paper_client: IPaperLookupClient = DiContainer().create_instance(
    spec={"di_factory": "lib/config/objects/ncbi_cache.yaml"}, resources={}
)


# %% Load the discrepancy list.

discrepancies = pd.read_csv(PIPELINE_ERROR_LIST, sep="\t")

# %% Load the error analysis outputs and consolidate into a single DataFrame that represents the "right" answer for each
# error.

consensus = pd.read_csv(CONSENSUS_RESOLUTION_FILE, sep="\t")
consensus.rename(columns={"Q###": "question_number"}, inplace=True)

ana_responses = [parse_error_analysis_excel(file) for file in ANALYST_RESPONSES_FILES]
ana_responses_cat = pd.concat(ana_responses)
ana_responses_cat.rename(columns={"Q###": "question_number"}, inplace=True)

# Groupby "question_number and aggregate "Response" as a list.
ana_responses_grouped = (
    ana_responses_cat.groupby(["question_number", "Phenotype"])
    .agg(
        Gene=("Gene", "first"),
        Paper=("Paper", "first"),
        Link=("Link", "first"),
        Response=("Response", list),
        Note=("Note", list),
    )
    .reset_index()
)

ana_responses_grouped["Response_Consensus"] = ana_responses_grouped["Response"].apply(
    lambda x: "MISSING" if len(set(x)) > 1 else x[0]
)

# Loop through everything in consensus, subbing in the value for Response_1
# into the right place in ana_responses_grouped.
consensus.fillna({"Phenotype_1": ""}, inplace=True)

for _, row in consensus.iterrows():
    q = row["question_number"]
    assert row["Response_1"] == row["Response_2"]

    # Find the row in ana_responses_grouped that matches this question_number and
    # phenotype.
    response_idx = (ana_responses_grouped["question_number"] == row["question_number"]) & (
        ana_responses_grouped["Phenotype"] == row["Phenotype_1"]
    )

    if response_idx.sum() == 0:
        print(f"Could not find {q} - {row["Phenotype_1"]} in ana_responses_grouped.")
        continue

    ana_responses_grouped.loc[response_idx, "Response_Consensus"] = row["Response_1"]

updates = ana_responses_grouped[["question_number", "Phenotype", "Gene", "Paper", "Response_Consensus"]]

# Make sure there are no missing values in Response_Consensus
if any(updates["Response_Consensus"] == "MISSING"):
    print("There are missing values in Response_Consensus. Please resolve before continuing.")
    print(updates.query("Response_Consensus == 'MISSING'"))
    raise ValueError("Missing values in Response_Consensus.")


# Collapse the phenotype responses into a single row for each question number where Response_Consensus is a list
# of all the individual responses.
def collapse_pheno_group(group_df: pd.DataFrame) -> pd.Series:
    if group_df.shape[0] == 1 and group_df.iloc[0]["Phenotype"] == "":
        return group_df.iloc[0]

    return pd.Series(
        {
            "question_number": group_df["question_number"].iloc[0],
            "Phenotype": "",
            "Gene": group_df["Gene"].iloc[0],
            "Paper": group_df["Paper"].iloc[0],
            # Response_Consensus is a dict mapping phenotype values to the response_consensus value
            "Response_Consensus": {row["Phenotype"]: row["Response_Consensus"] for _, row in group_df.iterrows()},
        }
    )


updates_collapsed = (
    updates.groupby("question_number")[updates.columns].apply(collapse_pheno_group).reset_index(drop=True)
)

# %% Annotate the discrepancies with the updated consensus values.

# Strip the Q prefix for question_number and turn it into an int
updates_collapsed["question_number"] = updates_collapsed["question_number"].str.replace("Q", "").astype(int)

# Merge the Response_Consensus values into the discrepancies DataFrame.
discrepancies.set_index("question_number", inplace=True)
updates_collapsed.set_index("question_number", inplace=True)
discrepancies["new_truth_value"] = updates_collapsed["Response_Consensus"]

# While we're at it, let's clean up the repeated values in truth_dict and output_dict.
for idx, row in discrepancies.iterrows():
    if row["task"] != "phenotype":
        continue
    discrepancies.at[idx, "truth_dict"] = {k: list(set(v)) for k, v in eval(row["truth_dict"]).items()}
    discrepancies.at[idx, "output_dict"] = {k: list(set(v)) for k, v in eval(row["output_dict"]).items()}


# %% Prepare an intermediate file for the v1.1 truthset to support the necessary manual curation of new observations.

new_obs = discrepancies.query(
    "task == 'observations' and truth_value == 'False' and new_truth_value == 'This is completely correct'"
).copy()

new_obs.drop(columns=["task", "truth_value", "new_truth_value", "index", "truth_dict", "output_dict"], inplace=True)


def get_link(pmid: str) -> str:
    paper = paper_client.fetch(pmid)
    if paper is None:
        return ""
    pmcid = paper.props.get("pmcid", "")
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"


new_obs["link"] = new_obs["pmid"].apply(get_link)

new_obs.to_csv("data/v1.1/new_observations_raw.tsv", sep="\t", index=False)


# %% In the sections below, we're going to make a checklist of all the changes that need to be made to the truthset.
#
#

# %% Load the original v1 truthset.

train_papers = pd.read_csv(f"{TRUTHSET_ROOT}/papers_train_v1.tsv", sep="\t")
test_papers = pd.read_csv(f"{TRUTHSET_ROOT}/papers_test_v1.tsv", sep="\t")

papers = pd.concat([train_papers, test_papers])

train_evidence = pd.read_csv(f"{TRUTHSET_ROOT}/evidence_train_v1.tsv", sep="\t")
test_evidence = pd.read_csv(f"{TRUTHSET_ROOT}/evidence_test_v1.tsv", sep="\t")

evidence = pd.concat([train_evidence, test_evidence])

group_assignments = pd.read_csv(f"{TRUTHSET_ROOT}/group_assignments.tsv", sep="\t")

discrepancies["group"] = discrepancies["gene"].map(group_assignments.set_index("gene")["group"])

# %% List the todos for papers, order by gene and group

print("#")
print("# PAPERS")
print("#")
print("")

paper_discrepancies = discrepancies.query("task == 'papers'")

for group, group_df in paper_discrepancies.groupby("group"):
    for gene, gene_df in group_df.groupby("gene"):
        # Papers that should be added (truth_value is False, new_truth_value is TRUE)
        add_papers = gene_df.query("truth_value == 'False' and new_truth_value == 'TRUE'")
        if add_papers.shape[0] != 0:
            print(f"Add papers for {gene} in group {group}: {', '.join(add_papers.pmid.astype(str).values)}")

        # Papers that should be removed (truth_value is True, new_truth_value is FALSE)
        remove_papers = gene_df.query("truth_value == 'True' and new_truth_value == 'FALSE'")
        if remove_papers.shape[0] != 0:
            print(f"Remove papers for {gene} in group {group}: {', '.join(remove_papers.pmid.astype(str).values)}")


# %% List the todos for observations, order by gene and group

# Note that the new_observations_raw.tsv file generated above should include all the new observations that need to
# be added, so here we're just going to list the observations that need to be removed.

print("#")
print("# OBSERVATIONS")
print("#")
print("")

print("New observations that need to be added can be found in data/v1.1/new_observations_raw.tsv")

obs_discrepancies = discrepancies.query("task == 'observations'")

for group, group_df in obs_discrepancies.groupby("group"):
    for gene, gene_df in group_df.groupby("gene"):
        # Observations that should be removed (truth_value is True, new_truth_value is anything other
        # than "This is completely correct")
        remove_obs = gene_df.query("truth_value == 'True' and new_truth_value != 'This is completely correct'")
        if remove_obs.shape[0] != 0:
            print(f"Remove observations for {gene} in group {group}: {'; ?'.join(remove_obs.id.values)}")


# %% List all the other content columns that need to be updated, order by group, gene, and id

print("#")
print("# CONTENT")
print("#")
print("")

content_discrepancies = discrepancies.query("task != 'papers' and task != 'observations'")

for group, group_df in content_discrepancies.groupby("group"):
    for gene, gene_df in group_df.groupby("gene"):
        print(f"{gene}")

        for id, id_df in gene_df.groupby("id"):
            for _, row in id_df.iterrows():
                if row["task"] == "phenotype":
                    # Any keys new_truth_value with a value of "TRUE" should be added if they're not already in truth.
                    adders = []
                    for k, v in row["new_truth_value"].items():
                        if v != "TRUE":
                            continue
                        hpo = k.split("(")[1].rstrip(")")
                        if any(hpo in truth_hpos for truth_hpos in row["truth_dict"].values()):
                            continue
                        adders.append(k)
                    if adders:
                        print(f"  Add HPO for {gene} in group {group} with id {id}: {', '.join(adders)}")

                    # Any keys new_truth_value with a value of "FALSE" should be removed if they're in truth.
                    for k, v in row["new_truth_value"].items():
                        if v != "FALSE":
                            continue
                        hpo = k.split("(")[1].rstrip(")")
                        if any(hpo in truth_hpos for truth_hpos in row["truth_dict"].values()):
                            print(f"  Remove HPO for {gene} in group {group} with id {id}: {k}")
                            break
                else:
                    if row["truth_value"].lower() != row["new_truth_value"].lower():
                        print(
                            f"  Update content for {gene} in group {group} with id {id}: {row['task']}: "
                            f"{row['truth_value']} -> {row['new_truth_value']}"
                        )

# %%
