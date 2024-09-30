#!/bin/bash

# Define the TRAIN and TEST sets of runs
TRAIN=(165847 210652 044027 134659 191020)
TEST=(165451 194240 223218 145606 181121)

# Loop through each element in the TRAIN list
for run_id in "${TRAIN[@]}"; do
    python scripts/benchmark/paper_finding/manuscript_paper_finding.py \
    -s "scripts/benchmark/paper_finding/paper_finding_benchmarks_skipped_pmids.txt" \
    -m data/v1/papers_train_v1.tsv -p "$run_id"
done

# Loop through each element in the TEST list
for run_id in "${TEST[@]}"; do
    python scripts/benchmark/paper_finding/manuscript_paper_finding.py \
    -s "scripts/benchmark/paper_finding/paper_finding_benchmarks_skipped_pmids.txt" \
    -m data/v1/papers_test_v1.tsv -p "$run_id"
done