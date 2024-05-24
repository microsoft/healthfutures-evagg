# %% Imports.

import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lib.evagg.llm import OpenAIClient
from lib.evagg.svc import get_dotenv_settings

# %% Setup.
# (evagg) azureuser@evagg-ash-1:~/ev-agg-exp$ python /home/azureuser/ev-agg-exp/notebooks/paper_finding_benchmarks.py -f="True" -l lib/config/paper_finding_benchmark.yaml -p=".out/final_isolated_92recall_paper_finding_results_2024-05-22_2aff4d256c1ab5e15e24378c95143f3992bcda3e/library_benchmark.tsv" > missed_papers_titles.txt # NOTE modified print statements to give me those outputs
# (evagg) azureuser@evagg-ash-1:~/ev-agg-exp$ grep "E.A. # Missed Papers:" .out/final_isolated_92recall_paper_finding_results_2024-05-22_2aff4d256c1ab5e15e24378c95143f3992bcda3e/benchmarking_paper_finding_results_train.txt

recall_92_missed = "/home/azureuser/ev-agg-exp/.out/final_isolated_92recall_paper_finding_results_2024-05-22_2aff4d256c1ab5e15e24378c95143f3992bcda3e/missed_papers_titles.txt"

recall_80_missed = "/home/azureuser/ev-agg-exp/.out/final_pipeline_80recall_paper_finding_results_2024-05-23_2aff4d256c1ab5e15e24378c95143f3992bcda3e/missed_papers_titles.txt"

# %% Build venn of misses between 92 and 80 recall files
# Load data
with open(recall_92_missed, "r") as f:
    next(f)
    missed_92 = f.readlines()

with open(recall_80_missed, "r") as f:
    next(f)
    missed_80 = f.readlines()

print("missed_92", missed_92)
import matplotlib.pyplot as plt

# %% Venn diagram of missed papers
from matplotlib_venn import venn2

# Convert lists to sets
set1 = set(missed_92)
set2 = set(missed_80)

venn = venn2([set1, set2], set_labels=("missed_isolated_run_92recall", "missed_pipeline_run_80recall"))

# Perform set operations
only_set1 = set1 - set2
only_set1 = {s.replace("\n", "") for s in only_set1}
only_set2 = set2 - set1
only_set2 = {s.replace("\n", "") for s in only_set2}
intersection = set1 & set2
intersection = {s.replace("\n", "") for s in intersection}

print(f"Only in missed_92: {only_set1}")
print(f"Only in missed_80: {only_set2}")
print(f"In both: {intersection}")

plt.show()


import matplotlib.pyplot as plt

# %% Create a wordcloud of the missed_80 papers and intersection papers
from wordcloud import WordCloud

# Convert the sets to strings
missed_80_str = " ".join(only_set2)
intersection_str = " ".join(intersection)

# Create word clouds
wordcloud_missed_80 = WordCloud().generate(missed_80_str)
wordcloud_intersection = WordCloud().generate(intersection_str)

# Display the word clouds
plt.imshow(wordcloud_missed_80, interpolation="bilinear")
plt.axis("off")
plt.show()

plt.imshow(wordcloud_intersection, interpolation="bilinear")
plt.axis("off")
plt.show()

# %% Show missed_80 and intersection papers
print("Missed papers in missed_80:")
print(only_set2)
print("\n")
print("Missed papers in both:")
print(intersection)


# %% What proportion of those missed_80 papers and intersection papers are in the content extraction truth set?
# Load the content extraction truth set
content_truth = "/home/azureuser/ev-agg-exp/data/v1/evidence_train_v1.tsv"

# Load the content extraction truth set
df = pd.read_csv(content_truth, sep="\t")

# Log any overlapping titles between the missed_80 and "paper_title" column in content_truth
overlap_missed_80 = df[df["paper_title"].isin(only_set2)].drop_duplicates(subset="paper_title")

# Do the same for intersection
overlap_intersection = df[df["paper_title"].isin(intersection)].drop_duplicates(subset="paper_title")

# Print the titles themselves in those two sets
print("Overlap missed_80:")
print(overlap_missed_80["paper_title"])
print("\n")
print("Overlap intersection:")
print(overlap_intersection["paper_title"])
print("\n")

len(overlap_missed_80)
len(overlap_intersection)

# %% Determine the difference in PMIDs between library_benchmark_rare_disease.tsv and the papers_train_v1_has_fulltext.tsv
lib_bench_rare_dis = "/home/azureuser/ev-agg-exp/.out/final_isolated_92recall_paper_finding_results_2024-05-22_2aff4d256c1ab5e15e24378c95143f3992bcda3e/library_benchmark_just_rare_disease.tsv"

papers_train_has_ft = "/home/azureuser/ev-agg-exp/.out/final_isolated_92recall_paper_finding_results_2024-05-22_2aff4d256c1ab5e15e24378c95143f3992bcda3e/papers_train_v1_has_fulltext.tsv"

# Load the two files
df_lib_bench_rare_dis = pd.read_csv(lib_bench_rare_dis, sep="\t", header=None)
print("df_lib_bench_rare_dis", df_lib_bench_rare_dis)

df_paper_train_has_ft = pd.read_csv(papers_train_has_ft, sep="\t", header=None)
print("df_paper_train_has_ft", df_paper_train_has_ft)


# %% Create a Venn diagram between the PMIDs in df_lib_bench_rare_dis and the PMIDs in df_paper_train_has_ft
# Get the PMIDs from the two dataframes
pmids_lib_bench_rare_dis = df_lib_bench_rare_dis[1].str.replace("pmid:", "")
set_pmids_lib_bench_rare_dis = set(pmids_lib_bench_rare_dis)
print("pmids_lib_bench_rare_dis", len(pmids_lib_bench_rare_dis), len(set_pmids_lib_bench_rare_dis))

# Print the difference in PMIDS between set(pmids_lib_bench_rare_dis) and pmids_lib_bench_rare_dis
# Convert the list to a set


# Find the PMIDs that are in set_pmids_list but not in set_pmids_lib_bench_rare_dis
diff_pmids = set_pmids_list.difference(set_pmids_lib_bench_rare_dis)

print("PMIDs in pmids_list but not in set_pmids_lib_bench_rare_dis:")
print(diff_pmids)


# %%
import pandas as pd

isolated_run = "/home/azureuser/ev-agg-exp/.out/final_isolated_92recall_paper_finding_results_2024-05-22_2aff4d256c1ab5e15e24378c95143f3992bcda3e/library_benchmark_just_rare_disease.tsv"
truth = pd.read_csv("data/v1/papers_train_v1.tsv", sep="\t")
output = pd.read_csv(".out/pipeline_benchmark.tsv", sep="\t", header=1)
# output = pd.read_csv(isolated_run, sep="\t")
# %%

truth_pmids = set(truth[truth.has_fulltext == True].pmid)
print("len(truth_pmids)", len(truth_pmids))
output_pmids = {int(r[1]) for r in output.paper_id.str.split(":")}
print("len(output_pmids)", len(output_pmids))

# Calculate the false positives
false_positives = output_pmids.difference(truth_pmids)
print("false_positives", len(false_positives))

false_negatives = truth_pmids.difference(output_pmids)
print("false_negatives", len(false_negatives))

true_positives = output_pmids.intersection(truth_pmids)
print("true_positives", len(true_positives))

# %%

fn = truth_pmids - output_pmids
print(len(truth_pmids), "-", len(output_pmids))
print("fn", len(fn))
fp = output_pmids - truth_pmids
print("fp", len(fp))

tp = truth_pmids & output_pmids
print("tp", len(tp))

precision = len(tp) / (len(tp) + len(fp))
recall = len(tp) / (len(tp) + len(fn))

print(f"Precision: {precision} (N_irrelevant = {len(fp)})")
print(f"Recall: {recall} (N_missed = {len(fn)})")

# %%
