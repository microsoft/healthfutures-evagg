# Setup

Evidence Aggregator runs at the Linux command line and depends on access to multiple required and optional external resources. The following document walks you through how to configure your local software and external cloud environment to perform a full-featured execution of an EvAgg [pipeline app](README.md#pipeline-apps).

**_These instructions have been tested on Ubuntu 20.04/22.04 in WSL2 and on Azure VMs_**

## Install software prerequisites

- **Python** 3.12 or above
- **miniconda** with libmamba solver

    ```bash
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > miniconda.sh
    sh ./miniconda.sh # close and reopen shell.
    /home/azureuser/miniconda3/bin/conda init $SHELL # if you didn't init conda for your shell during setup.
    conda update -n base -c defaults conda -y
    conda config --add channels conda-forge
    conda install -n base conda-libmamba-solver -y
    conda config --set solver libmamba
    ```

- **git**
- **make** [optional] only for [development tasks](README.md#pre-pr-checks)
- **jq** [optional] only for [development tasks](README.md#pre-pr-checks)
- **Azure CLI** [optional] only if using [AOAI RBAC authorization](#authentication-to-aoai)

  ```curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash```

## Clone the repository

Create and enter a local clone of this repository in your runtime environment:

```bash
git clone https://github.com/microsoft/healthfutures-evagg
cd healthfutures-evagg
```

## Build a conda environment

Create a conda environment. All shell commands in this section should be executed from the repository's root directory.

```bash
conda env create -f environment.yml
conda activate evagg
```

## Install poetry dependencies

Use poetry to install the local library and register the pipeline run command:

```bash
poetry install
```

Test installation by running the following. You should see a help message displayed providing usage for the `run_evagg_app` command.

```bash
run_evagg_app -h
```

## Deploy external resources

Evidence Aggregator may need to call out to various external resources (e.g. web endpoints or Azure services) during a given pipeline run. The following sections explain how these resources can be deployed and what settings are available to control how they are accessed during the run. All appropriate resource-specific configuration settings are provided to Evidence Aggregator at pipeline app runtime (as described below in [Configuration settings](#configuration-settings).

## Azure OpenAI (AOAI) Service

The only Azure resource strictly required to run the Evidence Aggregator pipeline is an AOAI Service instance, which can be created from within the Azure portal using [these instructions](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal). Once the AOAI Service is created, [deploy a model](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal#deploy-a-model) on it that has sufficient capabilities to run the example EvAgg pipeline app. Specifically, the pipeline currently requires a model that supports at least 128k tokens of input context and [json output mode](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/json-mode?tabs=python). Current models that meet these two requirements are `gpt-4` (`1106-preview` or newer), `gpt-4o`, and `gpt-4o-mini`. (See [this article](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models?tabs=python-secure) for a discussion of the differences between them.) Make sure you have sufficient quota (at least 100k TPM or equivalent) allocated to your model deployment.

### Authentication to AOAI

Authentication to the AOAI Service REST APIs can be done either using EntraID and role-based access control (RBAC) or shared keys. RBAC-based auth is recommended, but the EvAgg pipeline currently supports either.

- Option 1 - RBAC: In this mode, the pipeline uses the credentials of the logged-in user to authenticate to AOAI. When running the pipeline, ensure that you have run `az login` and have a current credential for an identity with (at minimum) the `Cognitive Services OpenAI User` role for your endpoint.
- Option 2 - Shared key: In this mode, the pipeline uses a key value provided at runtime to authenticate with AOAI. Make note of the shared key for your AOAI endpoint from within the Azure portal and supply it in the AOAI configuration settings.

Make note of the endpoint for AOAI Service Resource (typically `https://<your_resource_name>.openai.azure.com/`) and the name of the model deployment that you configured in the Azure Portal for use in configuration settings.

### Configuration settings for AOAI

- **AZURE_OPENAI_DEPLOYMENT** - set this to the model deployment name that you configured above (not necessarily the name of the model itself - the model deployment name is customizable during deployment configuration)
- **AZURE_OPENAI_ENDPOINT** - set this to the web endpoint for your AOAI Service Resource (typically "https://<your_resource_name>.openai.azure.com")
- **AZURE_OPENAI_API_KEY** [optional] - set this to one of your AOAI Service Resource's shared keys (only needed if you are using shared key-based authentication)
- **AZURE_OPENAI_API_VERSION** - set this to your AOAI version ("2024-02-15-preview" or later)
- **AZURE_OPENAI_MAX_PARALLEL_REQUESTS** [optional] - this controls the maximum number of concurrent outstanding requests being made to the AOAI endpoint - leave it out or set it to 0 (no max) unless you need to reduce the concurrency

## [Optional] Azure CosmosDB database

This pipeline makes a substantial number of reference lookup calls to external web resources, most notably NCBI's E-utilities service and the Mutalyzer service.
These calls can contribute substantially to pipeline execution time as access to those web services is rate limited in accordance with their usage guidelines. To speed up repeated execution of identical lookup calls to these services across pipeline runs, Evidence Aggregator can be configured to use an Azure CosmosDB instance as a results lookup cache keyed by the full URL/query string of the external service call.

1. Create an Azure CosmosDB account using [these instructions](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/quickstart-portal). For the most part, the default settings are likely suitable for your application, but you may apply/consider the following changes:
    1. Select a unique name for your CosmosDB account and make note of it as it will be used in subsequent configuration steps
    2. If your runtime environment is an Azure VM, you should consider deploying your CosmosDB Account in the same region
    3. Under capacity mode, switch the selection to "Serverless" as the sustained load for cache lookup calls is minimal
2. Within that new account, create a database and container using [these instructions](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/quickstart-portal#create-a-database-and-container). Use the following settings
    1. For the Database id, ensure that "Create new" is selected and enter `document_cache`
    2. For the Container id, enter `cache`
    3. For the Partition key, enter `/id`
3. If using EntraID-based authentication and authorization (recommended), configure role assignment settings for your CosmosDB account.
    - First, create a JSON file describing a cosmosdb SQL role definition.

    ```bash
    cat << EOF >> cache_role.json
    {
      "Id": "$(uuidgen)",
      "RoleName": "cache_role",
      "Type": "CustomRole",
      "AssignableScopes": ["/dbs/document_cache/colls/cache"],
      "Permissions": [{
        "DataActions": [
          "Microsoft.DocumentDB/databaseAccounts/readMetadata",
          "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/executeQuery",
          "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/read",
          "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/create",
          "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/replace",
          "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/upsert",
          "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/delete"
        ]
      }]
    }
    EOF
    ```

    - Next, create that role on your CosmosDB Account. Ensure that you are logged into `az cli` as an identity with sufficient privileges for this operation

    ```bash
    az cosmosdb sql role definition create -a <your comsosdb account> -g <your resource group> --body cache_role.json
    rm cache_role.json
    ```

    - Lastly, assign this role to the identity that will be running the pipeline

    ```bash
    az cosmosdb sql role assignment create -a <your cosmosdb account> -g <your resource group> \
      --role-definition-name "cache_role" \
      --scope "/dbs/document_cache/colls/cache" \
      --principal-id $(az ad signed-in-user show --query id --output tsv)
    ```

    Alternatively, if using shared key-based authentication, make note of the Primary Key for your CosmosDB account from within the Azure Portal for use in configuration settings.

### Configuration settings for CosmosDB

- **EVAGG_CONTENT_CACHE_ENDPOINT** - set this to the web endpoint for your CosmosDB instance (typically "https://<your_consmosdb_name>.documents.azure.com:443/")
- **EVAGG_CONTENT_CACHE_CREDENTIAL** [optional] - set this to the secret account key associated with your CosmosDB (only needed if you are using shared key-based authentication)
- **EVAGG_CONTENT_CACHE_CONTAINER** [optional] - set this to the container within your CosmosDB instance that contains the cache records (default: "cache")

## [Optional] NCBI E-utilities

The pipeline relies heavily on REST API calls to the NCBI E-utilities service, by default throttled on the service side to a maximum of 3 requests per minute. Pipeline execution can be accelerated by obtaining an API KEY: when the key is provided to E-utilities REST API calls, the max request rate is increased to 10 requests per minute.

See [this page](https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/) for additional documentation.

### Configuration settings (NCBI E-utilities)

- **NCBI_EUTILS_API_KEY** [optional] - an API KEY for the NCBI E-utilities web resource
- **NCBI_EUTILS_EMAIL** [optional] - an email address associated with this key

## Configuration settings

Once all required and optional external resources have been deployed, the appropriate configuration settings for each resource need to be made available to the pipeline runtime. The default way to do this is through the creation of a `.env` file at the root of your EvAgg repository containing the settings - at runtime, the `lib.evagg.utils.get_dotenv_settings` pipeline utility component can read in resource-specific settings from this file and provide them to the pipeline resource components that need them.

**Note: your `.env` file can potentially contain secrets and should not be committed to source control. The `.env` file is included in `.gitignore` to help prevent this from happening.**

There is a `template.env` file at the repo root that can be used as a starting point for configuring your own local `.env` file. Here is an minimal example `.env` file that can be used for a basic test of pipeline functionality if your AOAI Service Resource is configured for EntraID-based authentication/authorization.

```bash
AZURE_OPENAI_DEPLOYMENT="gpt-4-1106-preview" # whatever model name you chose during model deployment
AZURE_OPENAI_ENDPOINT="https://<your_aoai_resource>.openai.azure.com" # https endpoint for AOAI API calls
AZURE_OPENAI_API_VERSION="2024-02-15-preview"
```

As an alternative to `dotenv` file-based settings, the `lib.evagg.utils.get_env_settings` pipeline utility component is analogous to `get_dotenv_settings` and can be used to read configuration settings in directly from environment variables.

## Configure the pipeline app

The `lib/config/evagg_pipeline_example.yaml` app spec is a good starting place for your first full-featured execution of the pipeline. By default, this app is configured to use EntraID/RBAC auth for the AOAI Service and no caching for external web lookup calls. The configuration in the various `yaml` specs comprising the app can be modified before running to change the default behavior. For example:

 - **For key-based authentication to the AOAI Service:** Make sure **AZURE_OPENAI_API_KEY** is populated in your `.env` file with the appropriate key. Then comment out the `token_provider` section in each of `lib/config/objects/llm.yaml` and `lib/config/objects/llm_cache.yaml`. On `lib.evagg.llm.aoai.OpenAIClient` initialization as defined by these specs, this will leave the `token_provider` field of the `config` dictionary empty and force it to use shared key-based auth. In this case the `get_dotenv_settings` provider will have read the key in from the `.env` file and automatically populated the `api_key` field in `config`.
 - **For CosmosDB-based caching of external web lookup calls:** Make sure **EVAGG_CONTENT_CACHE_ENDPOINT** is populated in your `.env` file with the appropriate endpoint. Then, in `lib/config/evagg_pipeline_example.yaml`, uncomment the three commented-out lines that terminate in `_cache.yaml` and comment out each of their counterparts. This will swap out the non-caching `lib.evagg.utils.web.RequestsWebContentClient` implementers of `lib.evagg.utils.IWebContentClient` in the default web client sub-specs for the caching implementers in the `_cache.yaml` sub-specs. Each `_cache.yaml` sub-spec (e.g. `lib.config.objects.web_cache.yaml`) ultimately resolves to an instance of `lib.evagg.utils.web.CosmosCachingWebClient` that uses `get_dotenv_settings` to populate its endpoint/auth values from the **EVAGG_CONTENT_CACHE_** values in the `.env` file.
 - **For key-based authentication to the CosmosDB cache:** Make sure **EVAGG_CONTENT_CACHE_CREDENTIAL** is populated in your `.env` file with the appropriate key. Then comment out the `credential` section in `lib/config/objects/web_cache.yaml`. Much like the AOAI case above, on `lib.evagg.utils.web.CosmosCachingWebClient` initialization this will leave the `credential` field of the `cache_settings` dictionary empty and force it to use shared key-based auth. In this case the `get_dotenv_settings` provider will have read the key in from the `.env` file and automatically populated the `credential` field in `cache_settings`.

You can create your own pipeline apps by modifying sub-specs as necessary and assembling them into top-level app specs that can be run to produce the desired output with the required runtime configurations.

## Run the pipeline app

The script `run_evagg_app` is used to execute a pipeline app. It has one required argument - a pointer to an app spec `yaml` file that implements `lib.evagg.IEvAggApp` - and is invoked from the repository root. To execute the example app as configured above, run the following command. It will perform a PubMed query for a subset of potential papers with two query genes and write the resulting publication evidence table to file in the output directory. In a single test run, the pipeline identified 15 unique observations from a total of 4 publications considered in this configuration.

**_Note: If you are using RBAC-based auth to any Azure services, ensure that you have run `az login` and have a current credential for an identity that is authorized to perform the desired operations._**

```bash
run_evagg_app lib/config/evagg_pipeline_example.yaml
```

Using gpt-4o-mini with 2000k TPM of quota allocated, this example will complete in approximately 2 minutes. Results of the run will be located in `.out/run_evagg_pipeline_example_<YYYYMMDD_HHMMSS>`, where each individual run is given a unique datestamp suffix. The primary pipeline output file is `pipeline_benchmark.tsv` contained in that folder. Additional files in the run output folder can be used to debug issues and better understand interactions between the pipeline and the LLM.

You can optionally add or override any leaf value within an app spec dictionary (or sub-dictionary) using the `-o` argument followed by one or more dictionary `key:value` specifications separated by spaces. The following command overrides the default log level for increased logging verbosity and outputs the run results to a file named `different.tsv` in the run output directory:

```bash
run_evagg_app lib/config/evagg_pipeline_example.yaml -o writer.tsv_name:different log.level:DEBUG
```
