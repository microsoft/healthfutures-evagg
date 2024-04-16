#!/bin/bash
# paper_finding_prompt_assistant.sh

# Run the pipeline
echo "\nRunning paper finding pipeline ..."
#run_query_sync /home/azureuser/ev-agg-exp/lib/config/paper_finding_benchmark.yaml

# Run the benchmarks
echo "\nRunning benchmarks ..."
#python /home/azureuser/ev-agg-exp/notebooks/paper_finding_benchmarks.py -l lib/config/paper_finding_benchmark.yaml
tail -n 2 /home/azureuser/ev-agg-exp/.out/paper_finding_results_$(date +%Y-%m-%d)/benchmarking_paper_finding_results_train.txt

# Extract precision and recall values
precision=$(tail -n 3 /home/azureuser/ev-agg-exp/.out/paper_finding_results_$(date +%Y-%m-%d)/benchmarking_paper_finding_results_train.txt | head -n 3 | awk 'NR==2{print $3}')
recall=$(tail -n 2 /home/azureuser/ev-agg-exp/.out/paper_finding_results_$(date +%Y-%m-%d)/benchmarking_paper_finding_results_train.txt | head -n 2| awk 'NR==2{print $3}')

# Check if precision or recall is less than 1.0
if (( $(echo "$precision < 1.0" | bc -l) )) || (( $(echo "$recall < 1.0" | bc -l) )); then
  # Proceed to next step
  echo "Proceeding to next step..."

  # Run the llm_prompt_assistant
  echo "\nRunning LLM prompt assistant ..."
  python /home/azureuser/ev-agg-exp/notebooks/llm_prompt_assistant.py

  # Run the pipeline
  echo "\nRunning paper finding pipeline with updated prompts..."
  run_query_sync /home/azureuser/ev-agg-exp/lib/config/paper_finding_benchmark.yaml

  # Run the benchmarks
  echo "\nRunning benchmarks with updated prompts ..."
  python /home/azureuser/ev-agg-exp/notebooks/paper_finding_benchmarks.py -l lib/config/paper_finding_benchmark.yaml
  tail -n 2 /home/azureuser/ev-agg-exp/.out/paper_finding_results_$(date +%Y-%m-%d)/benchmarking_paper_finding_results_train.txt
else
  echo "Precision and recall are both 1.0. Exiting..."
  exit 0

fi