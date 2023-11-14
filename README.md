# Evidence Aggregation Experimentation Sandbox

This is an environment for experimentation on LLM prompts and related application code relevant to evidence aggregation for molecular diagnosis of mendelian disease.

## Compute environment pre-requisites

**Note: this environment has only been tested on Ubuntu 20.04 in WSL and an Azure VM. Behavior on other operating systems may vary. If you run into issues, please submit them on Github.**

The intended model for interacting with this environment is using VSCode. Many of the descriptions and instructions below will be specific to users of VSCode. Some of the command line functionality will work directly in the terminal.

Local Machine

- [Visual studio code](https://code.visualstudio.com/download)
- The Remote Development Extension Pack for VSCode.

Ubuntu 20.04 VM/WSL

- [Python](https://www.python.org/downloads/) 3.8 and above
- azcopy (install script)
- azure cli (install script)
- git (install script)
- make (install script)
- Miniconda

    ```bash
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > miniconda.sh
    sh ./miniconda.sh # close and reopen shell.
    export PATH=/home/azureuser/miniconda3/bin:$PATH
    conda update -n base -c defaults conda
    conda config --add channels conda-forge
    conda install -n base conda-libmamba-solver
    conda config --set solver libmamba
    ```

Open visual studio code on your local machine and connect to the VM using `Remote-SSH: Connect to Host`. This will initiate a connection to your VM within VSCode. Alternatively, if you're using WSL connect using `WSL: Connect to WSL`.

Next, clone this repository with `git clone https://github.com/jeremiahwander/ev-agg-exp`.

Open this newly cloned repository in VSCode with "File -> Open Folder"

You will be prompted to install recommended extensions for this repository, accept these recommendations.

## Environment setup

Create a conda environment. All shell commands in this section should be executed from the repository's root directory.

```bash
source activate base
conda env create -f environment.yml
conda activate evagg
```

Now use poetry to install the local library and registered scripts.

```bash
poetry install
```

Test installation of scripts by running the following command

```bash
run_query_sync -h
```

You should see a help message displayed providing usage for the `run_query_sync` script.

## Configuration

To communicate with various cloud resources, some additional configuration is necessary, including providing relevant 
secrets to the application.

Create a `.env` file in the repository root with the following format:

```text
AZURE_OPENAI_DEPLOYMENT_NAME=<your AOAI deployment name>
AZURE_OPENAI_ENDPOINT=<your AOAI endpoint>
AZURE_OPENAI_API_KEY=<your AOAI key>

NCBI_EUTILS_API_KEY=<your NCBI eutils API key (optional)>
NCBI_EUTILS_EMAIL=<your email address to be used for NCBI eutils API calls>
NCBI_EUTILS_MAX_TRIES=<max number of retries for retry-able calls to the NCBI eutils API (optional)>
```

## Running scripts

At this point, the only script that exists is `lib/scripts/run_query_sync`. It is fairly generic and can be used to run
a variety of apps, as long as they implement the protocol `lib.evagg.IEvaggApp`. Execution of the script is configured
entirely from within YAML files, examples for which are contained in `lib/scripts/config`. These files describe both the
wiring between dependencies and the parameterization of each dependency.

The script `run_query_sync` can be invoked as follows:

```bash
run_query_sync -c <path_to_config_file>
```

As a simple smoke test of script execution, one can run the following specific commands:

```bash
python sandbox/miah/make_local_library.py
run_query_sync -c lib/scripts/config/simple_config.py
```

This will create a fake local library of papers, extract content from those papers using a dummy implementation, and
print the results of these operations to stdout. While this doesn't actually do anything useful, it's a good way to
verify that the pipeline and dependencies are configured correctly.

For a more full-featured execution, after configuring your `.env` file with the correct secrets, you can run the
following:

```bash
mkdir -p .data/truth
azcopy login
azcopy cp "https://$SA.blob.core.windows.net/truth/truth_set_tiny.tsv" .data/truth/truth_set_tiny.tsv
run_query_sync -c lib/scripts/config/tiny_config.yaml
```

Where `$SA` is the Azure Blob Storage Account where your team stores test files.

## Contributing

### Code organization

The repository contains the following subdirectories:

```text
root
|-- deploy: infrastructure as code for the deployment and management of necessary cloud resources.
|-- docs: additional documentation beyond what is in scope for this README.
|-- lib: source code for scripts and core libraries.
|-- notebooks: shared notebooks for interactive development.
|-- sandbox: scratch area for developers.
|-- test: unit tests for core libraries.
```

### Questions and todos

- Consider writing script for literature library localization? Likely only necessary if we don't see ourselves moving directly to PMC API requests.
- Consider dataset organization, online access
- Consider base types for the results that we're pulling out of a paper, using primitives is ugly?
