#!/bin/bash

# Function to populate TRAIN and TEST arrays from JSON file
populate_arrays() {
    local model=$1
    local json_file=$2

    TRAIN=($(jq -r --arg model "$model" '.["runs"][] | select(.group == "train" and .model == $model) | .date' "$json_file"))
    TEST=($(jq -r --arg model "$model" '.["runs"][] | select(.group == "test" and .model == $model) | .date' "$json_file"))
}

# Check if model parameter is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <model>"
    exit 1
fi

MODEL=$1
JSON_FILE="scripts/benchmark/benchmark_runs.json"

# Populate TRAIN and TEST arrays
populate_arrays "$MODEL" "$JSON_FILE"

# Loop through each element in the TRAIN list
for run_id in "${TRAIN[@]}"; do
    echo "### Processing TRAIN run $run_id ###"
    python scripts/benchmark/paper_finding/manuscript_single_run.py \
    -s "scripts/benchmark/paper_finding/paper_finding_benchmarks_skipped_pmids.txt" \
    -m data/v1.1/papers_train_v1.1.tsv -p "$run_id"
done

# Loop through each element in the TEST list
for run_id in "${TEST[@]}"; do
    echo "### Processing TEST run $run_id ###"
    python scripts/benchmark/paper_finding/manuscript_single_run.py \
    -s "scripts/benchmark/paper_finding/paper_finding_benchmarks_skipped_pmids.txt" \
    -m data/v1.1/papers_test_v1.1.tsv -p "$run_id"
done