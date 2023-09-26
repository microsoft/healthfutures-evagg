# Evidence Aggregation Experimentation Sandbox

This is an environment for experimentation on LLM prompts and related application code relevant to evidence aggregation for molecular diagnosis of mendelian disease.

## Prerequisites, and environment setup

**Note: this environment has only been tested on Ubuntu 20.04. Behavior on other operating systems may vary. If you run into issues, please submit them on Github.**

The intended model for interacting with this environment is using VSCode. Many of the descriptions and instructions below will be specific to users of VSCode. Some of the command line functionality will work directly in the terminal.

Local Machine
- [Visual studio code](TODO)

Ubuntu 20.04 VM
- [Python](https://www.python.org/downloads/) 3.8 (TODO check version) and above
- Miniconda (TODO)

Open visual studio code on your local machine and connect to the VM using `Remote-SSH: Connect to Host`. This will initiate a connection to your VM within VSCode.

Next, clone this repository with `git clone https://github.com/jeremiahwander/ev-agg-exp`.

Open this newly cloned repository in VSCode with "File -> Open Folder"

You will likely be prompted to install recommended extensions for this repository, accept these recommendations.

Create a conda environment

```bash
conda env create -f environment.yml
conda activate evagg
```

## Configuration

## Running the application


## Development todo

- Support for chat API
- figure out what's going on with AOAI gpt-3.5-turbo (extended responses), probably because we're using the completions API
- tests for model code
- benchmarking
  - csv to test data
  - test data in bulk
  - comparing to truth labels
- how to embed the papers
- containerized / codespaces execution
- pythonpath in the shell, direnv?
- Conda or other dependency manager
- todo package version management