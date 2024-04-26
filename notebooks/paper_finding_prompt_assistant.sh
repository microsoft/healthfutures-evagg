#!/bin/bash
# paper_finding_prompt_assistant.sh

# Check if the first argument is "-h" or "--help"
if [[ $1 == "-h" || $1 == "--help" ]]; then
    echo "Usage: $0 [<max_iterations> <epsilon>]"
    echo "max_iterations: The maximum number of iterations to run. Default is 10."
    echo "epsilon: The minimum improvement in precision and recall to continue iterating. Default is 0.1."
    exit 0
fi

# Set the max iterations and epsilon from the command line arguments if not provided
max_iterations=${1:-10}  # Default max_iterations is 10
epsilon=${2:-0.1}  # Default epsilon is 0.1

echo "Running with max_iterations=$max_iterations and epsilon=$epsilon"

# Set the max iterations and epsilon from the command line arguments if not provided
max_iterations=${1:-3}  # Default max_iterations is 10
epsilon=${2:-0.1}  # Default epsilon is 0.1
 
echo -e "\n--> LLM prompt assistant (i.e. copilot^2) for paper finding ..."

# Initialize previous precision and recall
prev_precision=0
prev_recall=0

# Initialize counter
counter=0

# Add the date and git hash to the results directory
git_hash=$(git rev-parse HEAD)
dir_name="paper_finding_results_$(date +%Y-%m-%d)_${git_hash}"

# Run the pipeline
echo -e "\n--> Running paper finding pipeline ..."
run_query_sync lib/config/paper_finding_benchmark.yaml

# Run the benchmarks
echo -e "\n--> Running benchmarks ..."
python notebooks/paper_finding_benchmarks.py -l lib/config/paper_finding_benchmark.yaml

# Save the pipeline table output to the results directory
cp .out/library_benchmark.tsv .out/$dir_name

# Save the beginning/original/initial paper finding prompt to the results directory
cp lib/evagg/content/prompts/paper_finding.txt .out/$dir_name

# Extract precision and recall values
last_line=$(tail -n 1 .out/$dir_name/benchmarking_paper_finding_results_train.txt)
if [[ $last_line == "No true positives. Precision and recall are undefined." ]]; then
    echo "--> 0No true positives. Precision and recall are undefined."
else
    precision=$(tail -n 2 .out/$dir_name/benchmarking_paper_finding_results_train.txt | awk 'NR==1{print $3}')
    recall=$(tail -n 2 .out/$dir_name/benchmarking_paper_finding_results_train.txt | awk 'NR==2{print $3}')
    echo -e "\n--> Precision: $precision"
    echo -e "--> Recall: $recall"
fi

# Repeat until conditions are met $max_iterations times
while true; do

  # Check if precision or recall is less than 1.0
  if (( $(echo "$precision < 1.0" | bc -l) )) || (( $(echo "$recall < 1.0" | bc -l) )); then
    # Check if precision and recall have improved more than 0.1 
    if (( $(echo "$precision - $prev_precision > $epsilon" | bc -l) )) && (( $(echo "$recall - $prev_recall > $epsilon" | bc -l) )); then
      # Increment counter
      ((counter++))
      echo "here"

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
    last_line=$(tail -n 1 .out/$dir_name/benchmarking_paper_finding_results_train.txt)
    if [[ $last_line == "No true positives. Precision and recall are undefined." ]]; then
      echo "--> 1No true positives. Precision and recall are undefined."
    else
      echo -e "\n-->Precision and recall are both 1.0. Exiting..."
    fi
    exit 0
  fi
  
  # Extract precision and recall values
  last_line=$(tail -n 1 .out/$dir_name/benchmarking_paper_finding_results_train.txt)
  if [[ $last_line == "No true positives. Precision and recall are undefined." ]]; then
    echo "--> 2No true positives. Precision and recall are undefined."
  else
    # Update previous precision and recall
    prev_precision=$precision
    prev_recall=$recall
  fi 
done