# Benchmark Scripts #

This directory contains all the scripts necessary to recreate the benchmark analyses used in the v1 manuscript. If
necessary run outputs are localized, then the steps below can be followed to recreate the benchmark analyses.

## Pre-requisites ##

From the appropriate cloud storage location, localize the benchmark run outputs specified in `benchmark_runs.json`.

## Running benchmark analyses ##

**Note: all of the steps below assume your working directory is `<REPO_ROOT>`.


1. **Individual run analyses (multi-truthset support)**: 
   - The batch scripts (`batch_paper_finding.sh` and `batch_content_extraction.sh`) accept one or more truthset versions as additional arguments (default: `v1 v1.1`).
   - The scripts will process all specified truthset versions, passing the version to downstream scripts and preventing output overwrites by suffixing output files with the truthset version.
   - Example usage (processes both v1 and v1.1 by default):

    ```bash
    scripts/benchmark/content_extraction/batch_content_extraction.sh GPT-4-Turbo
    scripts/benchmark/paper_finding/batch_paper_finding.sh GPT-4-Turbo
    # To specify a single version:
    scripts/benchmark/content_extraction/batch_content_extraction.sh GPT-4-Turbo v1
    scripts/benchmark/paper_finding/batch_paper_finding.sh GPT-4-Turbo v1.1
    # To specify multiple versions explicitly:
    scripts/benchmark/content_extraction/batch_content_extraction.sh GPT-4-Turbo v1 v1.1
    scripts/benchmark/paper_finding/batch_paper_finding.sh GPT-4-Turbo v1 v1.1
    ```

   - The single-run analysis scripts (`manuscript_single_run.py` and `manuscript_obs_content_single_run.py`) now require a `--truthset-version` argument. All output files (not directories) will be suffixed with the specified truthset version.
   - Example usage:

    ```bash
    python scripts/benchmark/paper_finding/manuscript_single_run.py --truthset-version v1 ...
    python scripts/benchmark/content_extraction/manuscript_obs_content_single_run.py --truthset-version v1.1 ...
    ```

   - By default, outputs are placed in new, run-specific output folders in `.out`, with filenames indicating the truthset version.


2. **Figure generation (multi-truthset support)**: 
   - The figure generation scripts (except phenotype figures) now aggregate and plot results for all specified truthset versions, grouping bars by truthset version in the main barplots.
   - Run the following scripts in an interactive window or as python scripts (in this case outputs will be generated in new folders in the `.out/` directory):

    ```bash
    python scripts/benchmark/paper_finding/manuscript_generate_figures.py
    python scripts/benchmark/content_extraction/manuscript_generate_obs_figures.py
    python scripts/benchmark/content_extraction/manuscript_generate_content_figures.py
    ```

   - The first two scripts will also generate output TSVs for each truthset version, suffixed accordingly, for downstream analyses.
   - The main barplots will display grouped bars by truthset version for easy comparison.


3. **Discrepancy finding**: In an interactive window or as a python script, run
   - `scripts/benchmark/paper_finding/manuscript_discrepancy_list.py`,
   - `scripts/benchmark/content_extraction/manuscript_obs_discrepancy_list.py`, and
   - `scripts/benchmark/content_extraction/manuscript_content_discrepancy_list.py`.
   These scripts will generate TSVs for each truthset version (if applicable), written to `.out/manuscript_paper_finding` and `.out/manuscript_content_extraction`.

4. **Discrepancy question generation**: In an interactive window or as a python script, run
`scripts/benchmark/consolidate_discrepancy_questions.py`. This script will generate discrepancy questions and write out
a set of files that organize those questions by pmid for faster manual review. By default these output files will be
written to `.out/manuscript_content_extraction`.


5. **Model comparison**: In an interactive window (figures aren't currently written to disk), run
   - `scripts/benchmark/paper_finding/manuscript_model_comparison.py` and
   - `scripts/benchmark/content_extraction/manuscript_obs_model_comparison.py`
   to generate figures comparing various AOAI GPT models. Note: to run these notebooks you will need to have performed step 1 above for the pipeline outputs for all relevant truthset versions and models.

---

**Summary of multi-truthset-version workflow:**
- Batch scripts accept one or more truthset versions as arguments and process all specified versions.
- Single-run scripts require a `--truthset-version` argument and suffix output files with the version.
- Figure scripts (except phenotype figures) aggregate and plot results for all specified truthset versions, grouping bars by version.
- Output files and tables are version-suffixed to prevent overwrites and enable easy comparison.
