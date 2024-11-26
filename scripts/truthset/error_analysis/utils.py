import pandas as pd


def parse_error_analysis_excel(file_path: str) -> pd.DataFrame:
    """This function parses the error analysis Excel file and returns a DataFrame with the parsed data."""
    df = pd.read_csv(file_path, header=None, encoding="unicode_escape")

    genes = []
    papers = []
    links = []
    question_numbers = []
    questions_themselves = []
    all_responses = []
    all_notes = []
    all_phenotypes = []

    i = 0
    while i < len(df):
        if df.iloc[i, 0] == "Gene":
            gene = df.iloc[i, 1]
            paper = df.iloc[i + 1, 1]
            link = df.iloc[i + 2, 1]
            i += 3  # Move to the next row after the link

            while i < len(df) and df.iloc[i, 0] != "Gene":
                cell_value = str(df.iloc[i, 1]) if pd.notna(df.iloc[i, 1]) else ""
                if cell_value.startswith("Q"):
                    question_number, question_itself = cell_value.split(".", 1)

                    responses = []
                    phenotypes = []
                    optional_note = ""
                    j = i + 1
                    while j < len(df) and pd.notna(df.iloc[j, 2]):
                        if df.iloc[j, 2] == "[Optional] Note:":
                            optional_note = df.iloc[j, 3] if pd.notna(df.iloc[j, 3]) else ""
                            j += 1
                            continue
                        response = df.iloc[j, 2]
                        phenotype = df.iloc[j, 3] if pd.notna(df.iloc[j, 3]) else ""
                        responses.append(response)
                        phenotypes.append(phenotype)
                        j += 1

                    for response, phenotype in zip(responses, phenotypes):
                        genes.append(gene)
                        papers.append(paper)
                        links.append(link)
                        question_numbers.append(question_number)
                        questions_themselves.append(question_itself)
                        all_responses.append(response)
                        all_notes.append(optional_note)
                        all_phenotypes.append(phenotype)

                    i = j  # Move to the next question or gene
                else:
                    i += 1  # Move to the next row
        else:
            i += 1  # Move to the next row if not "Gene"

    # Create a DataFrame with the parsed data
    parsed_df = pd.DataFrame(
        {
            "Gene": genes,
            "Paper": papers,
            "Link": links,
            "Q###": question_numbers,
            "Questions": questions_themselves,
            "Response": all_responses,
            "Phenotype": all_phenotypes,
            "Note": all_notes,
        }
    )

    return parsed_df
