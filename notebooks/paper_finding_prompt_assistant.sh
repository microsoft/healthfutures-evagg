#!/bin/bash
# paper_finding_prompt_assistant.sh

# Initialize previous precision and recall
prev_precision=0
prev_recall=0

# Repeat up to 3 times
for i in {1..3}; do
  # Run the pipeline
  echo -e "\nRunning paper finding pipeline ..."
  run_query_sync /home/azureuser/ev-agg-exp/lib/config/paper_finding_benchmark.yaml

  # Run the benchmarks
  echo -e "\nRunning benchmarks ..."
  python /home/azureuser/ev-agg-exp/notebooks/paper_finding_benchmarks.py -l lib/config/paper_finding_benchmark.yaml

  # Extract precision and recall values
  precision=$(tail -n 2 /home/azureuser/ev-agg-exp/.out/paper_finding_results_$(date +%Y-%m-%d)/benchmarking_paper_finding_results_train.txt | awk 'NR==1{print $3}')
  recall=$(tail -n 2 /home/azureuser/ev-agg-exp/.out/paper_finding_results_$(date +%Y-%m-%d)/benchmarking_paper_finding_results_train.txt | awk 'NR==2{print $3}')

  # Check if precision or recall is less than 1.0
  if (( $(echo "$precision < 1.0" | bc -l) )) || (( $(echo "$recall < 1.0" | bc -l) )); then
    # Check if precision and recall have not changed more than 0.2
    if (( $(echo "$precision - $prev_precision < 0.2" | bc -l) )) && (( $(echo "$recall - $prev_recall < 0.2" | bc -l) )); then
      # Proceed to next step
      echo -e "Proceeding to next step..."

      # Run the llm_prompt_assistant
      echo -e "\nRunning LLM prompt assistant ..."
      python /home/azureuser/ev-agg-exp/notebooks/llm_prompt_assistant.py

      # Run the pipeline
      echo -e "\nRunning paper finding pipeline with updated prompts..."
      run_query_sync /home/azureuser/ev-agg-exp/lib/config/paper_finding_benchmark.yaml
    else
      echo -e "Precision and recall have changed more than 0.2. Exiting..."
      exit 0
    fi
  else
    echo -e "Precision and recall are both 1.0. Exiting..."
    exit 0
  fi

  # Update previous precision and recall
  prev_precision=$precision
  prev_recall=$recall
done