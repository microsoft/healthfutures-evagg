# Benchmark Scripts #

This directory contains all the scripts necessary to recreate the benchmark analyses used in the v1 manuscript. If
necessary run outputs are localized, then the steps below can be followed to recreate the benchmark analyses.

## Pre-requisites ##

From the appropriate cloud storage location, localize the benchmark run outputs specified in `benchmark_runs.json`.

## Running benchmark analyses ##

**Note: all of the steps below assume your working directory is `<REPO_ROOT>`.

1. **Individual run analyses**: This entails running `paper_finding/manuscript_single_run.py` and
`content_extraction/manuscript_obs_content_single_run.py` for every single output file localized. For the 30 runs
specified in `benchmark_runs.json` this could get a little tedious, so we've provided a shell script that will run
the above python scripts for all of the runs for a given model. These can be run as follows:

    ```bash
    scripts/benchmark/content_extraction/batch_content_extraction.sh GPT-4-Turbo
    scripts/benchmark/paper_finding/batch_paper_finding.sh GPT-4-Turbo
    scripts/benchmark/content_extraction/batch_content_extraction.sh GPT-4o
    scripts/benchmark/paper_finding/batch_paper_finding.sh GPT-4o
    scripts/benchmark/content_extraction/batch_content_extraction.sh GPT-4o-mini
    scripts/benchmark/paper_finding/batch_paper_finding.sh GPT-4o-mini
    ```

    If you are not planning on regenerating the model comparison outputs, then it is only necessary to run the first two
lines above. By default, the outputs for these analyses will be placed in new, run-specific output folders in `.out`.

2. **Figure generation**: In an interactive window (figures aren't currently written to disk), run
`scripts/benchmark/paper_finding/manuscript_generate_figures.py`,
`scripts/benchmark/content_extraction/manuscript_generate_figures.py`, and
`scripts/benchmark/content_extraction/content_benchmark.py`. These will generate a bunch of figures that can be used in
paper authoring. The first two scripts and will also generate a number of output TSVs that can be used in downstream
analyses.

3. **Discrepancy finding**: In an interactive window or as a python script, run
`scripts/benchmark/paper_finding/manuscript_discrepancy_list.py`,
`scripts/benchmark/content_extraction/manuscript_obs_discrepancy_list.py`, and
`scripts/benchmark/content_extraction/manuscript_content_discrepancy_list.py`. These scripts will generate TSVs that
will be used downstream in the generation of discrepancy questions. By default these TVS will be written to
`.out/manuscript_paper_finding` and `.out/manuscript_content_extraction`

4. **Discrepancy question generation**: In an interactive window or as a python script, run
`scripts/benchmark/consolidate_discrepancy_questions.py`. This script will generate discrepancy questions and write out
a set of files that organize those questions by pmid for faster manual review. By default these output files will be
written to `.out/manuscript_content_extraction`.

5. **Model comparison**: In an interactive window (figures aren't currently written to disk), run
`scripts/benchmark/paper_finding/manuscript_model_comparison.py` and
`scripts/benchmark/content_extraction/manuscript_obs_model_comparison.py` to generate figures comparing various AOAI 
GPT models. Note: to run these notebooks you will need to have performed step 1 above for the pipeline outputs for
GPT-4o and GPT4o-mini, not just GPT-4-Turbo.
