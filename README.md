# Evidence Aggregation Experimentation Sandbox

This is an environment for experimentation on LLM prompts and related application code relevant to evidence aggregation for molecular diagnosis of mendelian disease.

## Manual setup

The following sections walk through the manual setup of a Linux development environment, either a virtual machine or WSL2 install.

### Pre-requisites

**Note: this environment has only been tested on Ubuntu 20.04/22.04 in WSL and an Azure VM. Behavior on other operating systems may vary. If you run into issues, please submit them on Github.**

The intended model for interacting with this environment is using VSCode. Many of the descriptions and instructions below will be specific to users of VSCode. Some of the command line functionality will work directly in the terminal.

Local Machine

- [Visual Studio Code](https://code.visualstudio.com/download)
- The Remote Development Extension Pack for VSCode.

Ubuntu 20.04/22.04 VM/WSL

- [Python](https://www.python.org/downloads/) 3.8 and above
- azcopy (install script)
- azure cli (install script)
- git (install script)
- make (install script)
- docker (install script, only necessary if you're going to develop with devcontainers/codespaces)
- miniconda

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

Open this newly cloned repository in VSCode with **File -> Open Folder**

You will be prompted to install recommended extensions for this repository; accept these recommendations.

### Configuring an environment

Create a conda environment. All shell commands in this section should be executed from the repository's root directory.

```bash
source activate base
conda env create -f environment.yml
conda activate evagg
```

Now use poetry to install the local library and register scripts.

```bash
poetry install
```

Test script installation of by running the following command:

```bash
run_query_sync -h
```

You should see a help message displayed providing usage for the `run_query_sync` script.

## Codespaces setup

The following sections walk through the setup of a GitHub Codespaces-based development environment. In order to do this
you will need a GitHub account.

### Starting an environment

Open [the evagg repository](https://github.com/jeremiahwander/ev-agg-exp) in a web browser and use these instructions to
[create a new codespace with options](https://docs.github.com/en/codespaces/developing-in-a-codespace/creating-a-codespace-for-a-repository).

When selecting options, you should choose a suitable branch from the repository (likely `main`), and a region where you
want to do your work. As currently architected, the evagg pipeline doesn't directly interact with region-specific cloud
resources, but this may change in the future. A 2-core machine is currently suitable for pipeline execution. You will want
to select `evagg` as your devcontainer configuration.

Initial build of the codespaces environment can take a few minutes.

### Stopping an environment

By default, codespaces environments will stop automatically after 30 minutes of inactivity. If you want to manually stop
a codespace navigate to [codespaces management](https://github.com/codespaces) and manually stop the instance.

### Managing costs

See this document on [codespaces costs](https://docs.github.com/en/billing/managing-billing-for-github-codespaces/about-billing-for-github-codespaces)
for how to manage costs and billing for github codespaces usage. This repository is currently set up for user payment for
codespaces. Github personal free accounts include 120 core hours and 15 GiB per month.

## Running experiments

The library code consists of a variety of independent components for querying, filtering, and aggregating genetic
variant publication evidence. These components are designed to be assembled in various ways into an experiment "app"
that can be run via the `run_query_sync` entrypoint to perform some sort of concrete analysis.

### Defining experiment apps

An "app" is any Python class that implements the `lib.evagg.IEvAggApp` protocol (essentially just an `execute` method).
An app is defined in a yaml specification file as an ordered dictionary of key/value pairs describing the full
class/parameter hierarchy to be instantiated before `execute` is called. The abstract format of a yaml spec
is as follows:

```yaml
resource_1: <value or spec sub-dictionary>
...
resource_n: <value or spec sub-dictionary>

di_factory: <fully-qualified class name, factory method, or yaml file path>
param_1: <value or spec sub-dictionary>
...
param_n: <value or spec sub-dictionary>
```

The `lib.di` module is responsible for parsing, instantiating, and returning an arbitrary app object from an app spec.
For a given spec, it first collects, in order, the top-level name/value pairs - "resources" are entries occurring
before the `di_factory` key, and "parameters" are those occurring after. It then resolves any string value in the
parameter collection of the form `"{{resource_name}}"` by looking `resource_name` up in the "resources" collection -
this allows one to define a set of singleton objects earlier in the spec that may be reused at multiple places later
in the spec. Finally, the `di` module instantiates and returns the top-level object represented by the spec by invoking
the `di_factory:` entrypoint value, passing in the resolved/collected parameters as named keyword arguments. If any
resource or parameter value in the spec consists of a sub-dictionary with its own `di_factory` key, that value is first
resolved to a runtime object, recursively, following the same mechanism just described for the top-level spec. Object
hierarchies of this sort may be arbitrarily deep.

A library of existing app specs for defining various experiments can be found in `lib/config/`.

### Executing experiment apps

The script `run_query_sync` is used to execute an experiment app. It has one required argument - a pointer to a yaml
app spec - and is invoked as follows:

```bash
run_query_sync <path_to_yaml_app_spec_file>
```

This will instantiate the app object via the `lib.di` system described above and call the `IEvAggApp.execute`
method on it.

**Note:** A yaml spec may define any instantiable object, but `run_query_sync` assumes a spec defining a top-level
object that implements `IEvAggApp`.

As a simple test of script execution, run the following specific commands from the repo root:

```bash
python test/make_local_library.py
run_query_sync lib/config/simple_config.yaml
```

This will create a fake local library of papers, extract content from those papers using a dummy implementation, and
print the results of these operations to `stdout`. While this doesn't actually do anything useful, it's a good way to
verify that the pipeline and dependencies are configured correctly.

You can optionally add or override any leaf value within an app spec dictionary or sub-dictionary using
the `-o` argument followed by one or more dictionary key:value specifications separated by a space:

```bash
run_query_sync lib/config/table_config.yaml -o writer.outputpath:.out/override.tsv log.level:DEBUG
```

### Configuration and secrets

Various concrete components in the library code (e.g. `OpenAIClient`, `NcbiLookupClient`) require runtime
configuration and/or secrets passed in as constructor arguments in order to securely access external resources. This is
done via settings dictionaries that are instantiated with an appropriate factory specified in the app spec yaml.
The most common built in factory is `lib.evagg.svc.get_dotenv_settings`, which reads all settings with a given
prefix (e.g. `AZURE_OPENAI_`) from a `.env` file in the repo root and parses them into a corresponding settings
dictionary. A template file `template.env` documenting known component settings is offered at the repo root - to get
started, copy this file , rename to `.env`, and fill it in with actual secrets as needed.

For development on this repo from within [Codespaces](#codespaces-setup),
make use of the analogous factory `lib.evagg.svc.get_env_settings` in the app spec, which reads
settings in from environment variables. Create the required GitHub secrets and settings by following
[these instructions](https://docs.github.com/en/codespaces/managing-your-codespaces/managing-secrets-for-your-codespaces)
and granting the ev-agg-exp repo access to those secrets.

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
