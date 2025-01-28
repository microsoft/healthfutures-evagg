"""Estimate cost differences based on model.

This is a quick script to generate cost estimates for running the pipeline using
different AOAI models on the dev and eval datasets.
"""

# %% Imports.

import yaml

# %% Constants.

TRAIN_GENE_PATH = "lib/config/queries/mgttrain_subset.yaml"
TEST_GENE_PATH = "lib/config/queries/mgttest_subset.yaml"

MEAN_TRAIN_M_TOKENS = sum([39.5, 38.8, 38.7, 40.5]) / 4
MEAN_TEST_M_TOKENS = sum([22.6, 21.9, 23.5, 21.4, 22.4]) / 5

MODEL_COSTS_PER_M_TOKENS = {"GPT-4-Turbo": 10, "GPT-4o": 2.5, "GPT-4o-mini": 0.15}

# %% Load data.

train = yaml.safe_load(open(TRAIN_GENE_PATH))
test = yaml.safe_load(open(TEST_GENE_PATH))

# %%

train_papers = sum([g["retmax"] for g in train])
test_papers = sum([g["retmax"] for g in test])

# %% Compute the number of tokens per thousand papers returned by pubmed.

train_tokens_per_thousand = MEAN_TRAIN_M_TOKENS / (train_papers / 1000)
test_tokens_per_thousand = MEAN_TEST_M_TOKENS / (test_papers / 1000)

# On the order of 15M tokens per thousand papers.

# %% Compute cost estimates.

print("Expected cost estimates per 1000 papers returned by pubmed:")
for model, cost_per_m_tokens in MODEL_COSTS_PER_M_TOKENS.items():
    train_cost = train_tokens_per_thousand * cost_per_m_tokens
    test_cost = test_tokens_per_thousand * cost_per_m_tokens
    print(f"{model} ${train_cost:.2f}-${test_cost:.2f}")


# %%

print("Expected costs for full analysis of 47000 papers.")
for model, cost_per_m_tokens in MODEL_COSTS_PER_M_TOKENS.items():
    train_cost = train_tokens_per_thousand * cost_per_m_tokens * 47
    test_cost = test_tokens_per_thousand * cost_per_m_tokens * 47
    print(f"{model} ${train_cost:.2f}-${test_cost:.2f}")

# %%
