"""This notebook is intended to be used for analysis of user study actions.

This includes generation of paper figures and statistics about actions and action timing from the user study.
"""

# %% Imports.

import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import mixedlm

# %% Constants.

ACTIONS_ROOT = "data/user_study"
ACTIONS_FILES = [action_file for action_file in os.listdir(ACTIONS_ROOT) if action_file.endswith(".csv")]

# %% Function definitions.


def interpret_action_file_name(filename: str) -> pd.Series:
    """Interpret the filename of an action file."""
    parts = filename.split("_")
    assert len(parts) == 4
    return pd.Series(
        {
            "filename": filename,
            "participant_id": parts[0],
            "session_id": parts[1],
            "case_group_id": parts[2],
        }
    )


# %% Build a dataframe mapping action files to their metadata.

action_files_df = pd.DataFrame([interpret_action_file_name(action_file) for action_file in ACTIONS_FILES])

# %% Load the action data

# Each action file is a CSV with a single line header and the following columns, "action", "timestamp", and "notes".
# "action" and "notes" are free text strings, "timestamp" is a string formatted m/d/yyyy HH:mm:ss

actions_list: List[pd.DataFrame] = []

for _, row in action_files_df.iterrows():

    action_data = pd.read_csv(os.path.join(ACTIONS_ROOT, row.filename))

    # Convert the timestamp to a datetime.
    action_data["timestamp"] = pd.to_datetime(action_data["timestamp"])

    # Add the metadata to the action data.
    action_data["participant_id"] = row.participant_id
    action_data["session_id"] = row.session_id
    action_data["case_group_id"] = row.case_group_id

    actions_list.append(action_data)

# Concatenate the action data.
actions = pd.concat(actions_list)

# Order the action data session_id and participant_id.
actions = actions.sort_values(["participant_id", "session_id"])

# %% Collect counts of different types of actions in a dataframe indexed by participant_id and session_id.


def count_action(action_names: List[str]) -> pd.DataFrame | pd.Series:
    """Count the number of actions in the actions dataframe."""
    return actions.query("action in @action_names").groupby(["participant_id", "session_id"]).size()


case_review_counts = count_action(["start case", "completed case set- no additional variants for review"]) - 1
variant_review_counts = count_action(["new variant review", "review saved variant"])
pub_search_counts = count_action(["publication search"])
pub_read_counts = count_action(["publication reading"])
table_use_counts = count_action(["table interaction"])

action_counts = pd.DataFrame(
    {
        "case_review": case_review_counts,
        "variant_review": variant_review_counts,
        "pub_search": pub_search_counts,
        "pub_read": pub_read_counts,
        "table_use": table_use_counts,
    }
).fillna({"case_review": 0, "variant_review": 0, "pub_search": 0, "pub_read": 0})

# Annotate the rows of action_counts by adding case_group_id as a column, using the metadata from action_files_df.
action_counts = action_counts.reset_index().merge(
    action_files_df.drop(columns=["filename"]), on=["participant_id", "session_id"]
)

# %% Collect time durations for different action pairs, indexed by participant_id and session_id.


def duration_for_actions(start_actions: List[str], end_actions: List[str]) -> pd.DataFrame:
    if any(a in start_actions for a in end_actions) or any(a in end_actions for a in start_actions):
        print(
            f"Warning: start_actions and end_actions overlap: {start_actions} {end_actions}. "
            + "Subsequent warnings may be unreliable."
        )

    durations_map = {}

    for id, group_df in actions.groupby(["participant_id", "session_id", "case_group_id"]):
        start_row = None
        duration_list = []
        # Iterate through the rows of group_df, looking for start_actions and end_actions.
        for i, row in group_df.iterrows():

            # If we're in a window and we can close it, do so.
            if row.action in end_actions and start_row is not None:
                duration_list.append((row.timestamp, start_row.timestamp))
                start_row = None
            # If we're not in a window and we try to close one, warn.
            elif row.action in end_actions and start_row is None:
                print(f"Warning: found end action while not in window: {id} / {i}")

            # If we're not in a window (could be the same action where we just closed a window), and we can open one, do so.
            if row.action in start_actions and start_row is None:
                start_row = row
            # If we're in a window and we try to open one, warn.
            elif row.action in start_actions and start_row is not None:
                print(f"Warning: found start action while already in window: {id} / {i}")

        if start_row is not None:
            print(f"Info: found start action without end action: {id} / {i}")

        durations_map[id] = duration_list

    # Convert the durations_map into a dataframe, with the columns "participant_id", "session_id", and "duration". One row for every duration in the list.
    durations_list = []
    for id, durations in durations_map.items():
        for duration in durations:
            durations_list.append(
                {
                    "participant_id": id[0],
                    "session_id": id[1],
                    "case_group_id": id[2],
                    "start": duration[1],
                    "end": duration[0],
                    "duration": duration[0] - duration[1],
                }
            )

    durations_df = pd.DataFrame(durations_list)

    durations_df["seconds"] = durations_df["duration"].dt.total_seconds()
    durations_df.drop(columns=["duration"], inplace=True)

    return durations_df


print("-- Searching for case_review_durations --")
case_review_durations = duration_for_actions(
    ["start case"], ["start case", "completed case set- no additional variants for review"]
)

print("-- Searching for variant_review_durations --")
variant_review_durations = duration_for_actions(
    ["new variant review", "review saved variant"], ["move on from variant"]
)

# %% Make counts performance barplots.

# First, make a barplot with a pair of bars for each individual, stratified by session.
sns.set_theme(style="whitegrid")

ylabels = {
    "case_review": "Cases Reviewed",
    "variant_review": "Variants Reviewed",
    "pub_search": "Publication Searches",
    "pub_read": "Publications Read",
    "table_use": "Table Interactions",
}
for col in ylabels.keys():

    plt.figure()
    sns.barplot(data=action_counts, x="session_id", y=col, errorbar="sd", hue="case_group_id")
    plt.ylabel(ylabels[col])

    plt.figure()
    sns.barplot(data=action_counts, y=col, x="participant_id", hue="session_id", palette="pastel")
    plt.ylabel(ylabels[col])

# %% Make duration barplots.

case_review_durations["minutes"] = case_review_durations["seconds"] / 60

plt.figure()

sns.barplot(data=case_review_durations, x="session_id", y="minutes", errorbar="sd", hue="case_group_id")
plt.ylabel("Time spent on each case (minutes)")

plt.figure()
sns.barplot(data=case_review_durations, y="minutes", x="participant_id", hue="session_id", palette="pastel")
plt.ylabel("Time spent on each case (minutes)")

variant_review_durations["minutes"] = variant_review_durations["seconds"] / 60

plt.figure()
sns.barplot(data=variant_review_durations, x="session_id", y="minutes", errorbar="sd", hue="case_group_id")
plt.ylabel("Time spent on each variant (minutes)")

plt.figure()
sns.barplot(data=variant_review_durations, y="minutes", x="participant_id", hue="session_id", palette="pastel")
plt.ylabel("Time spent on each variant (minutes)")

# %% Perform statistical analyses of the counts.

for col in ylabels.keys():
    if col == "table_use":
        continue
    print(f"-- {ylabels[col]} --")
    model = mixedlm(f"{col} ~ session_id + case_group_id", action_counts, groups=action_counts["participant_id"])
    result = model.fit()
    print(result.summary())

# # Mixed effects model for duration data
print("-- Case review durations --")
model = mixedlm(
    "seconds ~ session_id + case_group_id", case_review_durations, groups=case_review_durations["participant_id"]
)
result = model.fit()
print(result.summary())

print("-- Variant review durations --")
model = mixedlm(
    "seconds ~ session_id + case_group_id", variant_review_durations, groups=variant_review_durations["participant_id"]
)
result = model.fit()
print(result.summary())

# %% Intentionally empty.
