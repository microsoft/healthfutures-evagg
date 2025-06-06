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
    echo "Usage: $0 <model> [truthset_version ...]"
    exit 1
fi

MODEL=$1
shift
TRUTHSET_VERSIONS=("$@")
if [ ${#TRUTHSET_VERSIONS[@]} -eq 0 ]; then
    TRUTHSET_VERSIONS=(v1 v1.1)
fi
JSON_FILE="scripts/benchmark/benchmark_runs.json"

for TRUTHSET_VERSION in "${TRUTHSET_VERSIONS[@]}"; do
    echo "Processing for truthset version: $TRUTHSET_VERSION"
    # Populate TRAIN and TEST arrays
    populate_arrays "$MODEL" "$JSON_FILE"

    # Loop through each element in the TRAIN list
    for run_id in "${TRAIN[@]}"; do
        echo "### Processing TRAIN run $run_id for $TRUTHSET_VERSION ###"
        python scripts/benchmark/paper_finding/manuscript_single_run.py \
        -s "scripts/benchmark/paper_finding/paper_finding_benchmarks_skipped_pmids.txt" \
        -m "data/$TRUTHSET_VERSION/papers_train_$TRUTHSET_VERSION.tsv" -p "$run_id" \
        --truthset-version "$TRUTHSET_VERSION"
    done

    # Loop through each element in the TEST list
    for run_id in "${TEST[@]}"; do
        echo "### Processing TEST run $run_id for $TRUTHSET_VERSION ###"
        python scripts/benchmark/paper_finding/manuscript_single_run.py \
        -s "scripts/benchmark/paper_finding/paper_finding_benchmarks_skipped_pmids.txt" \
        -m "data/$TRUTHSET_VERSION/papers_test_$TRUTHSET_VERSION.tsv" -p "$run_id" \
        --truthset-version "$TRUTHSET_VERSION"
    done
done