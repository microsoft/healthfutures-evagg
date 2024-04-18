#!/bin/bash
# paper_finding_prompt_assistant.sh

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <max_iterations> <epsilon>"
    exit 1
fi

# Get the max iterations and epsilon from the command line arguments
max_iterations=$1
epsilon=$2
 
echo -e "\n-->LLM prompt assistant (i.e. copilot^2) for paper finding ..."

# Initialize previous precision and recall
prev_precision=0
prev_recall=0

# Initialize counter
counter=0

# Repeat until conditions are met $max_iterations times
while true; do

  # Run the pipeline
  echo -e "\n-->Running paper finding pipeline ..."
  run_query_sync lib/config/paper_finding_benchmark.yaml

  # Run the benchmarks
  echo -e "\n-->Running benchmarks ..."
  python notebooks/paper_finding_benchmarks.py -l lib/config/paper_finding_benchmark.yaml

  # Save the pipeline table output to the results directory
  cp .out/library_benchmark.tsv .out/paper_finding_results_$(date +%Y-%m-%d)

  # Save the beginning/original/initial paper finding prompt to the results directory
  cp lib/evagg/content/prompts/paper_finding.txt .out/paper_finding_results_$(date +%Y-%m-%d)

  # Extract precision and recall values
  precision=$(tail -n 2 .out/paper_finding_results_$(date +%Y-%m-%d)/benchmarking_paper_finding_results_train.txt | awk 'NR==1{print $3}')
  recall=$(tail -n 2 .out/paper_finding_results_$(date +%Y-%m-%d)/benchmarking_paper_finding_results_train.txt | awk 'NR==2{print $3}')
  echo -e "\n-->Precision: $precision"
  echo -e "-->Recall: $recall"

  # Check if precision or recall is less than 1.0
  if (( $(echo "$precision < 1.0" | bc -l) )) || (( $(echo "$recall < 1.0" | bc -l) )); then
    # Check if precision and recall have improved more than 0.1 
    if (( $(echo "$precision - $prev_precision > $epsilon" | bc -l) )) && (( $(echo "$recall - $prev_recall > $epsilon" | bc -l) )); then
      # Increment counter
      ((counter++))

      # Check if counter has reached max_iterations
      if ((counter == max_iterations)); then
        echo -e "\n-->Precision: $precision"
        echo -e "-->Recall: $recall"
        echo -e "\n-->Precision and recall have changed $max_iterations times. Exiting..."
        exit 0
      fi

      # Proceed to next step
      echo -e "\n-->Proceeding to next step..."

      # Run the llm_prompt_assistant
      echo -e "\n-->Running LLM prompt assistant ..."
      set -a; source .env; set +a;
      python notebooks/llm_prompt_assistant.py

      # Run the pipeline
      echo -e "\n-->Running paper finding pipeline with updated prompts..."
      run_query_sync lib/config/paper_finding_benchmark.yaml
    else
      echo -e "\n-->Precision: $precision"
      echo -e "-->Recall: $recall"
      echo -e "\n-->Precision and recall have not improved more than 0.1. Exiting..."
      exit 0
    fi
  else
    echo -e "\n-->Precision and recall are both 1.0. Exiting..."
    exit 0
  fi

  # Update previous precision and recall
  prev_precision=$precision
  prev_recall=$recall
done