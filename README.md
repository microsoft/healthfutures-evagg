# Evidence Aggregation Experimentation Sandbox

This is an environment for experimentation on LLM prompts and related application code relevant to evidence aggregation for molecular diagnosis of mendelian disease.

## Compute environment pre-requisites

**Note: this environment has only been tested on Ubuntu 20.04 in WSL and an Azure VM. Behavior on other operating systems may vary. If you run into issues, please submit them on Github.**

The intended model for interacting with this environment is using VSCode. Many of the descriptions and instructions below will be specific to users of VSCode. Some of the command line functionality will work directly in the terminal.

Local Machine
- [Visual studio code](TODO)
- The Remote Development Extension Pack for VSCode.

Ubuntu 20.04 VM/WSL
- [Python](https://www.python.org/downloads/) 3.8 and above
- azcopy (TODO)
- azure cli (TODO)
- git (TODO)
- Miniconda (TODO)
- Poetry (curl -sSL https://install.python-poetry.org | python3 -)

Open visual studio code on your local machine and connect to the VM using `Remote-SSH: Connect to Host`. This will initiate a connection to your VM within VSCode. Alternatively, if you're using WSL connect using `WSL: Connect to WSL`.

Next, clone this repository with `git clone https://github.com/jeremiahwander/ev-agg-exp`.

Open this newly cloned repository in VSCode with "File -> Open Folder"

You will be prompted to install recommended extensions for this repository, accept these recommendations.

## Environment setup

Create a conda environment. All shell commands in this section should be executed from the repository's root directory.

```bash
conda env create -f environment.yml
conda activate evagg
```

Now use poetry to install the local library and registered scripts.

```bash
poetry install
```

Test installation of scripts by running the following command

```bash
run_single_query_sync -h
```

You should see a help message displayed providing usage for the `run_single_query_sync` script.

## Configuration

At this point, no additional configuration is needed, when we start using the Azure OpenAI service and/or other cloud services, we will need to put additional configuration detail here.

## Code organization

The repository contains the following subdirectories:

root
 - deploy: infrastructure as code for the deployment and management of necessary cloud resources.
 - docs: additional documentation beyond what is in scope for this README.
 - lib: source code for scripts and core libraries.
 - notebooks: shared notebooks for interactive development.
 - test: unit tests for core libraries.

## Running Benchmarks

### run_single_query_sync.py

`run_single_query_sync.py` is intended to be a standalone, synchronous benchmark for the full evidence aggregation pipeline. Given a single query (i.e., gene-variant pair, accepted nomenclatures TBD), this script will attempt to find all papers relevant to the query and extract a set of structured fields for all relevant variants within those papers.

As currently implemented, this script is pretty dumb. It leverages a local library of papers backed by one or more directories of json files, each one represnting a paper (see data dependencies (link TODO)). When asked for papers relevant to a query, all papers in the library are returned.

Further, this script leverages an implementation of content extraction that has a fixed set of fields of interest (e.g., MOI, phenotype) and returns a static value for each of those fields for exactly one variant in each paper.

Once the set of structured content has been extracted from the paper, it can either be written to disk or displayed as console output, as an exercise to demonstrate how dependencies are injected into the app.

#### Data pre-requisites

Execution of this script depends on a local library of papers, which can be downloaded from (TODO storage location) using the following commands.

```
sudo mkdir -p -m=0777 /mnt/data
azcopy login # This will log in to your default tenant, otherwise use --tenant-id
azcopy cp -r "https://<SA>.blob.core.windows.net/library/tiny_positive/" /mnt/data (TODO verify syntax)
```

If you place the local paper library in a different location, note you will need to modify `lib/scripts/run_single_query_sync/config/example_config.json` accordingly for `run_single_query_sync` to be able to run. By default the above commands will localize these files to the temp disk on an Azure VM; the temp disk is ephemeral storage and the data must be re-localized each time the VM is restarted.

#### Running the script

This script is configured in pyproject.toml as an installed script. It can be run using the following command, executed from the repository root directory:

```bash
run_single_query_sync -c lib/evagg/scripts/run_single_query_sync/config/example_config.json
```

Successful execution should result in the following output and a JSON file should be written to the `.out` directory in your repository root.

```
Writing output to: .out/run_single_query_sync.json
```

#### Evaluating results

No implementation of comparing this pipeline's output to ground truth has been implemented yet, but when one exists, it will probably exist in notebooks.

## Questions and todos
- Consider specifying injected dependencies in config files? Seems like a PITA, but will ultimately be much more flexible.
- Consider writing script for literature library localization? Likely only necessary if we don't see ourselves moving directly to PMC API requests.
- TODO: Linting env, vscode dependencies, and readme
- TODO: Actually doing the linting
- TODO: VSCode Interactive dependencies and setup 
- TODO: pytest env dependencies
- TODO: implement basic tests
- TODO: data science env dependencies
- TODO: notebook for processing the current literature spreadsheet
- TODO: dataset organization, and readme
- Consider base types for the results that we're pulling out of a paper?