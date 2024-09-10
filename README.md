# Evidence Aggregator

This repo contains the code and deployment instructions for the Evidence Aggregator, a collaboration between the
[Broad Institute](https://www.broadinstitute.org/) and [Microsoft Research Health Futures](https://www.microsoft.com/en-us/research/lab/microsoft-health-futures/).
This project aims to support genomic analysts in diagnosis of potential Mendelian disease by using Large Language Models to review external public literature databases (i.e. PubMed) and summarize the rare disease-relevant information for a given case. The code implements a set of Python modules configurable into a pipeline for running experiments using Azure OpenAI endpoints and producing output tables with clinically-relevant aggregated publication evidence for input gene or genes of interest.

## Setup

The following sections walk through the setup of a Linux development environment on a virtual machine or within WSL2.
Alternatively, there is a `.devcontainer` defined at the repo root for use with
[GitHub Codespaces](https://docs.github.com/en/codespaces/developing-in-a-codespace/creating-a-codespace-for-a-repository).

### Pre-requisites

**Note: this environment has been tested on Ubuntu 20.04/22.04 in WSL2 and Azure VMs.**

Local Machine

- [Visual Studio Code](https://code.visualstudio.com/download)
- The Remote Development Extension Pack for VSCode.

Ubuntu 20.04/22.04 (VM/WSL2)

- [Python](https://www.python.org/downloads/) 3.12 and above
- azcopy
- azure cli
- git
- make
- miniconda:

    ```bash
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > miniconda.sh
    sh ./miniconda.sh # close and reopen shell.
    /home/azureuser/miniconda3/bin/conda init $SHELL # if you didn't init conda for your shell during setup.
    conda update -n base -c defaults conda -y
    conda config --add channels conda-forge
    conda install -n base conda-libmamba-solver -y
    conda config --set solver libmamba
    ```

Open VSCode on your local machine and connect to the VM using `Remote-SSH: Connect to Host`. This will initiate a
connectionto your VM within VSCode. Alternatively, if you're using WSL connect using `WSL: Connect to WSL`. Next,
clone this repository with `git clone https://github.com/microsoft/healthfutures-evagg`. Open the newly-cloned
repository in VSCode with `File -> Open Folder`. You will be prompted to install recommended extensions for this
repository; accept these recommendations.

### Configuring an environment

Create a conda environment. All shell commands in this section should be executed from the repository's root directory.

```bash
conda env create -f environment.yml
conda activate evagg
```

Use poetry to install the local library and register pipeline run endpoint.

```bash
poetry install
```

Test endpoint installation by running the following command:

```bash
run_evagg_app -h
```

You should see a help message displayed providing usage for the `run_evagg_app` endpoint.

## Running the pipeline

The library code consists of a variety of independent pipeline components for querying, filtering, and aggregating genetic
variant publication evidence. These components are designed to be assembled in various ways into a pipeline "app"
that can be run via the `run_evagg_app` entrypoint to generate some sort of concrete analysis or output.

### Defining pipeline apps

An "app" is any Python class that implements the `lib.evagg.IEvAggApp` protocol (effectively an `execute` method).
An app is defined in a yaml specification file as an ordered dictionary of key/value pairs describing the full
class/parameter component hierarchy to be instantiated before `execute` is called. The abstract format of a yaml spec
is as follows:

```yaml
# Reusable resource definitions.
resource_1: <value or spec sub-dictionary>
...
resource_n: <value or spec sub-dictionary>

# Application definition.
di_factory: <fully-qualified class name, factory method, or yaml file path>
param_1: <value or spec sub-dictionary>
...
param_n: <value or spec sub-dictionary>
```

The `lib.di` module is responsible for parsing, instantiating, and returning an arbitrary app object from an app spec.
For a given spec, it first collects, in order, the top-level name/value pairs - "resources" are entries occurring
before the `di_factory` key, and "parameters" are those occurring after. It then resolves any string value in the
parameter collection of the form `"{{resource_name}}"` by looking `resource_name` up in the "resources" collection -
this allows instantiation of singleton objects earlier in the spec that may be reused at multiple places later
in the spec. Finally, the `di` module instantiates and returns the top-level app object represented by the spec by invoking
the `di_factory:` entrypoint value, passing in the resolved/collected parameters as named keyword arguments. If any
resource or parameter value in the spec consists of a sub-dictionary with its own `di_factory` key, that value is first
resolved to a runtime object, recursively, following the same mechanism just described for the top-level spec. Object
hierarchies of this sort may be arbitrarily deep.

A library of existing app and sub-object specs for defining various pipelines can be found in `lib/config/`.

### Executing pipeline apps

The script `run_evagg_app` is used to execute a pipeline app. It has one required argument - a pointer to a yaml
app spec - and is invoked as follows:

```bash
run_evagg_app <path_to_yaml_app_spec_file>
```

This will instantiate the app object via the `lib.di` system described above and call the `IEvAggApp.execute`
method on it. For convenience you may omit the the `lib/config/` prefix or the `.yaml` suffix.

**Note:** A yaml spec may define any instantiable object, but `run_evagg_app` requires a spec defining a top-level
object that implements `IEvAggApp`.

As a simple test of script execution, run the following command from the repo root:

```bash
# equivalently: run_evagg_app sample_config
run_evagg_app lib/config/sample_config.yaml
```

This extracts content from a local sample library of fake papers using a dummy implementation and prints the resulting
evidence table to `stdout`. While this doesn't actually do anything useful, it's a good way to verify that the
pipeline and its dependencies are configured correctly.

You can optionally add or override any leaf value within an app spec dictionary (or sub-dictionary) using
the `-o` argument followed by one or more dictionary `key:value` specifications separated by spaces. The following
command overrides the default log level and outputs the run results to a named file in the `.out` directory:

```bash
run_evagg_app sample_config -o writer.tsv_name:sample_test log.level:DEBUG
```

### Configuration and secrets

Various concrete components in the library code (e.g. `OpenAIClient`, `NcbiLookupClient`) require runtime
configuration and/or secrets passed in as constructor arguments in order to securely access external resources. This is
done via settings dictionaries that are instantiated with an appropriate factory specified in the app spec yaml.
The most common built in factory is `lib.evagg.utils.get_dotenv_settings`, which reads all settings with a given
prefix (e.g. `AZURE_OPENAI_`) from a `.env` file in the repo root and parses them into a corresponding settings
dictionary. A template file `template.env` documenting known component settings is offered at the repo root. To get
started, copy this file, rename it to `.env`, and fill it in with actual values as needed.

For development on this repo from within [Codespaces](#codespaces-setup),
make use of the analogous factory `lib.evagg.utils.get_env_settings` in the app spec, which reads
settings in from environment variables. Create the required GitHub secrets and settings by following
[these instructions](https://docs.github.com/en/codespaces/managing-your-codespaces/managing-secrets-for-your-codespaces)
and granting the repo access to those secrets.

## Contributing

### Code organization

The repository contains the following subdirectories:

```text
root
|-- deploy: infrastructure as code for the deployment and management of necessary cloud resources
|-- lib: source code for scripts and core libraries
|-- notebooks: shared notebooks for interactive development
|-- test: unit tests for core libraries
```

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.
