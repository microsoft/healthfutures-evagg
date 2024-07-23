# Scripts

This directory is the location for shared scripts relevant to EvAgg. These scripts should be runnnable in the base
environment with no additional configuration. If there are data dependencies for a given script, these dependencies
and how to obtain them should be clearly documented.

## Subdirectories

Scripts are logically separated by their role in the development and evaluation process for EvAgg. Below is an 
explanation of this division:

- `archive`: Currently unused scripts retained for reference. These are not maintained and are therefore not guaranteed
to run.
- `benchmark`: Scripts used for benchmarking pipeline outputs against a truth set. Scripts in this folder are used to 
generate statistics, tables, and figures used in the initial EvAgg publication(s).
- `comparison`: Scripts used for comparing pipeline outputs to the SOTA. Scripts in this folder are used to generate 
statistics, tables, and figures used in the initial EvAgg publication(s).
- `post_processing`: Scripts used for post-processing EvAgg outputs in preparation for loading into seqr as a reference
resource.
- `prompt_engineering`: Scripts used for the modification/optimization of prompts used in the primary EvAgg pipeline.
Currently only `lib/evagg/library/prompts/paper_category.txt` is modified by scripts in this directory.
- `truthset`: Scripts used for the review, preprocessing, and arbitration of truthsets used in benchmarking.
- `user_study`: Scripts used in the preparation for and execution of the EvAgg user study.
