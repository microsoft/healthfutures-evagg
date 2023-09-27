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