# %%
import pandas as pd

# File paths for the Excel files containing the discrepancy resolution data
file_path_analyst1 = "data/discrepancy_resolution/ana1_discrepancy_resolution.10.23.24.xlsx"
file_path_analyst2 = "data/discrepancy_resolution/ana2_discrepancy_resolution_completed_24Oct2024.xlsx"
file_path_analyst3 = "data/discrepancy_resolution/ana3_discrepancy_resolution_23Oct2024.xlsx"


# %% Parse the Excel file containing the discrepancy resolution data
def parse_discrepancy_excel(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path, header=None)

    # Initialize lists to store the parsed data
    genes = []
    papers = []
    links = []
    question_numbers = []
    questions_themselves = []
    responses = []
    notes = []

    # Iterate through the DataFrame to extract the required information
    for i in range(len(df)):
        if df.iloc[i, 0] == "Gene":
            gene = df.iloc[i, 1]
            paper = df.iloc[i + 1, 1]
            link = df.iloc[i + 2, 1]
            question_number = df.iloc[i + 4, 1].split(".")[0]
            question_itself = df.iloc[i + 4, 1].split(".")[1]
            response_cell = df.iloc[i + 5, 2]
            note_cell = df.iloc[i + 6, 3] if pd.notna(df.iloc[i + 6, 3]) else ""

            # Check if the response cell is not empty
            if pd.notna(response_cell):
                response = response_cell
            else:
                response = ""

            genes.append(gene)
            papers.append(paper)
            links.append(link)
            question_numbers.append(question_number)
            questions_themselves.append(question_itself)
            responses.append(response)
            notes.append(note_cell)

    # Create a DataFrame with the parsed data
    parsed_df = pd.DataFrame(
        {
            "Gene": genes,
            "Paper": papers,
            "Link": links,
            "Q###": question_numbers,
            "Questions": questions_themselves,
            "Response": responses,
            "Note": notes,
        }
    )

    return parsed_df


# Example usage
parsed_df_analyst1 = parse_discrepancy_excel(file_path_analyst1)
parsed_df_analyst2 = parse_discrepancy_excel(file_path_analyst2)
parsed_df_analyst3 = parse_discrepancy_excel(file_path_analyst3)


# %%
# Between the three dataframes, return a subset of the Q### and Questions where the "Response" columns show differing results on the same Q###, and have a new Notes_1 and Notes_2 column that show the notes for each of the differing results.
# Write a function to return that subset of the table
import pandas as pd


def compare_discrepancy_dfs(df1, df2):
    # Merge the two dataframes on the "Q###" column
    merged_df = df1.merge(df2, on="Q###", suffixes=("_1", "_2"))

    # Filter the merged dataframe to include only rows where the responses are different
    filtered_df = merged_df[merged_df["Response_1"] != merged_df["Response_2"]]

    # Create a subset of the filtered dataframe with the required columns
    subset_df = filtered_df[["Q###", "Gene_1", "Paper_1", "Link_1", "Response_1", "Note_1", "Response_2", "Note_2"]]

    # TODO: Fix the multi embedded rows for e.g. phenotype and extracting the associated columns.
    return subset_df.reset_index(drop=True)


# Example usage
# Assuming parsed_df_analyst1, parsed_df_analyst2, and parsed_df_analyst3 are already defined
subset_df_1_2 = compare_discrepancy_dfs(parsed_df_analyst1, parsed_df_analyst2)
subset_df_1_3 = compare_discrepancy_dfs(parsed_df_analyst1, parsed_df_analyst3)
subset_df_2_3 = compare_discrepancy_dfs(parsed_df_analyst2, parsed_df_analyst3)

print("Discrepancies between Analyst 1 and Analyst 2:")
print(subset_df_1_2)
print("\nDiscrepancies between Analyst 1 and Analyst 3:")
print(subset_df_1_3)
print("\nDiscrepancies between Analyst 2 and Analyst 3:")
print(subset_df_2_3)

# Save the discrepancies to .out files
subset_df_1_2.to_csv(".out/discrepancies_analyst1_vs_analyst2.tsv", index=False, sep="\t")
subset_df_1_3.to_csv(".out/discrepancies_analyst1_vs_analyst3.tsv", index=False, sep="\t")
subset_df_2_3.to_csv(".out/discrepancies_analyst2_vs_analyst3.tsv", index=False, sep="\t")
# %%
