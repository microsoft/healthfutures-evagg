#!/bin/bash
# copilot_squared.sh
# Ashley Conard
# This script is in the buisness of finding the best prompts for the paper finding task.
# First, it runs the paper finding pipeline with the original prompts.
# Second, it runs the paper finding benchmarks.
# Third, if the precision and recall are not both 1.0, it runs the LLM prompt assistant to hopefully improve the prompts
# Fourth, it runs the paper finding pipeline and benchmarks again.
# Fifth, it checks if the precision and recall have improved more than epsilon (after the first min_iterations iterations), and will stop after max_iterations iterations.

# Defaults
DEFAULT_MIN_ITERATIONS=3
DEFAULT_MAX_ITERATIONS=6
DEFAULT_EPSILON=0.1
DEFAULT_YAML="lib/config/paper_finding_benchmark.yaml"
DEFAULT_OUT_TSV=".out/library_benchmark.tsv"

# Functions
usage() {
    echo "Usage: $0 <min_iterations> <max_iterations> <epsilon> <yaml>"
    echo "min_iterations: The minimum number of iterations to run. Default is $DEFAULT_MIN_ITERATIONS."
    echo "max_iterations: The maximum number of iterations to run. Default is $DEFAULT_MAX_ITERATIONS."
    echo "epsilon: The minimum improvement in precision and recall to continue iterating. Default is $DEFAULT_EPSILON."
    echo "yaml: The yaml file for Evidence Aggregator pipeline to use. Default is $DEFAULT_YAML."
    echo "tsv: The tsv file for Evidence Aggregator pipeline to use. Default is $DEFAULT_OUT_TSV."
}

error() {
    echo "Error: $1"
    usage
    exit 1
}

run_pipeline() {
    echo -e "\n-->Running paper finding pipeline with updated prompts..."
    run_evagg_app $DEFAULT_YAML
}

run_benchmarks() {
    echo -e "\n--> Running benchmarks ..."
    python scripts/benchmark/paper_finding_benchmarks.py -l $DEFAULT_YAML
}

run_prompt_assistant() {
    echo -e "\n-->Running Copilot^2 assistant ..."
    set -a; source .env; set +a;
    python scripts/prompt_engineering/copilot_squared_assistant.py
}

save_pipeline_output_extract_precision_recall() {
    local last_line=$1
    local results_directory_name=$2
    local iteration=$3

    # Save the pipeline table output to the results directory
    cp $DEFAULT_OUT_TSV .out/$results_directory_name/paper_finding_benchmark_${iteration}_$(date +%H:%M:%S).tsv

    if [[ $last_line == "No true positives. Precision and recall are undefined." ]]; then
        echo "--> No true positives. Precision and recall are undefined. Setting both to 0."
        prev_precision=0
        prev_recall=0
    else
        precision=$(tail -n 2 .out/$results_directory_name/benchmarking_paper_finding_results_train.txt | awk 'NR==1{print $3}')
        recall=$(tail -n 2 .out/$results_directory_name/benchmarking_paper_finding_results_train.txt | awk 'NR==2{print $3}')
        echo -e "\n--> Precision: $precision"
        echo -e "--> Recall: $recall"
    fi

    # Set/'return' global variables
    PRECISION=$precision
    RECALL=$recall
}

# Main script
if [[ $1 == "-h" || $1 == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -gt 1 && $# -ne 5 ]]; then
    error "If more than one argument is provided, all arguments must be present."
fi

min_iterations=${1:-$DEFAULT_MIN_ITERATIONS}
max_iterations=${2:-$DEFAULT_MAX_ITERATIONS}
epsilon=${3:-$DEFAULT_EPSILON}

git_hash=$(git rev-parse HEAD)

echo -e "\n--> Copilot^2 for paper finding ..."
echo "Running with min_iterations=$min_iterations, max_iterations=$max_iterations, epsilon=$epsilon, yaml=$DEFAULT_YAML, and tsv=$DEFAULT_OUT_TSV.
Results will be saved in .out/paper_finding_results_$(date +%Y-%m-%d)_${git_hash}."

prev_precision=0
prev_recall=0
counter=0

# Output directory
results_directory_name="paper_finding_results_$(date +%Y-%m-%d)_${git_hash}"
mkdir -p $results_directory_name

# Run pipeline and benchmarks
run_pipeline
run_benchmarks

# Save the orignal prompt to the results directory
cp lib/evagg/content/prompts/paper_finding_few_shot.txt .out/$results_directory_name

# Repeat until conditions are met $max_iterations times
while true; do
  # Extract precision and recall values
  last_line=$(tail -n 1 .out/$results_directory_name/benchmarking_paper_finding_results_train.txt)
  save_pipeline_output_extract_precision_recall "$last_line" "$results_directory_name" "$counter"

  # Check if precision and recall are both 1.0
  if (( $(echo "$precision == 1.0" | bc -l) )) && (( $(echo "$recall == 1.0" | bc -l) )); then
    echo -e "\n-->Precision and recall are both 1.0. Exiting..."
    break
  fi

  # Increment counter
  ((counter++))

  # Check if counter has reached max_iterations
  if ((counter > max_iterations)); then
    echo -e "\n--> Max iterations reached."
    echo "--> Precision: $precision"
    echo "--> Recall: $recall"
    break
  fi

  # Check if counter is less than or equal to min_iterations
  if ((counter <= min_iterations)); then
    echo -e "\n--> Iteration: $counter/$max_iterations"
    echo "|precision - prev_precision|, |recall - prev_recall|"
    echo "$precision" - "$prev_precision", "$recall" - "$prev_recall"

    # Update previous precision and recall
    prev_precision=$precision
    prev_recall=$recall

    # Run the copilot_squared_assistant, pipeline, and benchmarks
    run_prompt_assistant
    run_pipeline
    run_benchmarks

  else
    # Check if precision and recall have improved more than epsilon
    if (( $(echo "sqrt(($precision - $prev_precision)^2) > $epsilon" | bc -l) )) && (( $(echo "sqrt(($recall - $prev_recall)^2) > $epsilon" | bc -l) )); then
      echo -e "\n--> Iteration: $counter/$max_iterations"
      echo "|precision - prev_precision|, |recall - prev_recall|"
      echo "$precision" - "$prev_precision", "$recall" - "$prev_recall"

      # Update previous precision and recall
      prev_precision=$precision
      prev_recall=$recall

      # Run the copilot_squared_assistant, pipeline, and benchmarks
      run_prompt_assistant
      run_pipeline
      run_benchmarks

    else
      echo "--> Precision and recall have not improved more than $epsilon. Exiting..."
      break
    fi
  fi

done

# Change results directory name to add copilot_squared_ at the beginning
mv .out/$results_directory_name .out/copilot_squared_$results_directory_name/
echo "Results are stored in .out/copilot_squared_${results_directory_name}/"