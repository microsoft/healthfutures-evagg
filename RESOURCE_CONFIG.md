# Resource configuration

Evidence Aggregator depends on access to multiple required external resources and optional external resources, for example the Azure OpenAI (AOAI) Resource Service. Some of these external dependencies can be used without secrets or authentication, while others require them. The following sections discuss how to configure the pipeline for access to these external resources.

## Overview - external dependencies

The following steps will allow you to run a test execution of the pipeline that verifies full functionality. They assume that you have completed the software environment setup in [README](README.md).

1. [AOAI](#deploy-and-configure-an-aoai-service-resource) - Deploy an AOAI Service Resource, deploy a gpt-4 (or equivalent) model, and optionally configure role-based authorization (recommended)
2. [Optional] [CosmosDB caching](#optional-deploy-and-configure-an-azure-cosmosdb-database) - Deploy and configure a CosmosDB account and database
3. [Optional] [NCBI E-utilities](#optional-obtain-an-api-key-for-the-ncbi-e-utilities-resource) - Obtain an API key for using NCBI's E-utilities
4. [.env](#configure-a-local-env-file) - Configure `template.env` with your external dependency settings and save as `.env`
5. [Optional] [Modify the pipeline config](#modify-the-example-pipeline-configuration) - Modify `evagg_pipeline_example.yaml` based on choices made above
6. [Run the pipeline](#running-the-pipeline) - Run the pipeline with the example configuration (`run_evagg_app lib/config/evagg_pipeline_example.yaml`)

After pipeline execution is complete, results can be viewed in `.out/run_evagg_pipeline_example_<YYYYMMDD_HHMMSS>`

## Deploy and configure an AOAI Service resource

The only Azure Resource required to run the Evidence Aggregator pipeline is an Azure OpenAI Service Resource which can be deployed using [these instructions](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal).
Once the AOAI Service Resource is deployed, it is necessary to [deploy a model](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal#deploy-a-model) on that resource that has sufficient capabilities to run the EvAgg pipeline. Specifically, the pipeline currently requires that a model supports at least 128k tokens of input context and the [json output mode](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/json-mode?tabs=python). Currently, the models that meet these two requirements are `gpt-4` (`1106-preview` or newer), `gpt-4o`, and `gpt-4o-mini`. See [this article](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models?tabs=python-secure) for a discussion of the differences between these models. Ensure that one of these models is deployed in your AOAI Service Resource and that you have sufficient quota (at least 100k TPM or equivalent) allocated to that deployment.

Authentication/authorization to use the AOAI Service Resource REST APIs can be done either using EntraIDs and role-based access control (RBAC) or shared keys. RBAC-based authorization is recommended, but the EvAgg pipeline currently supports either. After deploying your AOAI Service Resource you will either need to configure RBAC for the identity that will be running the pipeline or take note of one of the shared keys for your AOAI Service Resource. The minimum privilege built-in RBAC role needed to use the AOAI REST APIs during pipeline execution is the `Cognitive Services OpenAI User` role.

Additionally, you will need to make note of the endpoint for AOAI Service Resource (typically `https://<your_resource_name>.openai.azure.com/`) and the name of the model deployment that you configured.

## [Optional] Deploy and configure an Azure CosmosDB database

This pipeline makes a substantial number of calls to external web resources, most notably NCBI's E-utilities service and the Mutalyzer service.
These calls can contribute substantially to pipeline execution time as access to those web services is rate limited in accordance with their
usage guidelines, however the responses are deterministic, so making repeated identical calls to rate limited services can decrease performance.
In particular for repeated executions of similarly configured pipelines (e.g., development scenarios), it can substantially increase performance
to enable Azure-local caching of responses from these web services.

To avoid repeated execution of identical external calls to these external services, Evidence Aggregator can be configured to use a database-based cache mechanism. This depends on the availability of a correctly configured Azure CosmosDB instance.

1. Create an Azure CosmosDB account using [these instructions](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/quickstart-portal). For the most part, the default settings are likely suitable for your application, but apply/consider the following changes
    1. Select a unique name for your CosmosDB account and make note of it as it will be used in subsequent configuration steps
    2. If your runtime environment is an Azure VM, you should consider deploying your Cosmos DB Account in the same region
    3. Under capacity mode, switch the selection to "Serverless" as the sustained loads for DBs in this account are likely to be minimal
2. Within that new account, create a database and container using [these instructions](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/quickstart-portal#create-a-database-and-container). Use the following settings
    1. For the Database id, ensure that "Create new" is selected and enter `document_cache`
    2. For the Container id, enter `cache`
    3. For the Partition key, enter `/id`
3. If using EntraID-based authentication and authorization (recommended) configure role assignment settings for your CosmosDB account.
    1. First, create a JSON file describing a cosmosdb SQL role definition.

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

    2. Now, create that role on your CosmosDB Account. Ensure that you are logged into `az cli` as an identity with sufficient privileges for this operation

    ```bash
    az cosmosdb sql role definition create -a <your comsosdb account> -g <your resource group> --body cache_role.json
    rm cache_role.json
    ```

    3. Lastly, assign this role to the identity that will be running the pipeline

    ```bash
    az cosmosdb sql role assignment create -a <your cosmosdb account> -g <your resource group> \
      --role-definition-name "cache_role" \
      --scope "/dbs/document_cache/colls/cache" \
      --principal-id $(az ad signed-in-user show --query id --output tsv)
    ```

    Alternatively, if using shared key-based authentication, make note of the Primary Key for your CosmosDB account from within the Azure Portal

## [Optional] Obtain an API key for the NCBI E-utilities resource

The pipeline relies heavily on REST API calls to the NCBI E-utilities service. By default, these requests are throttled to 3 requests per minute. Pipeline execution
can be accelerated by obtaining an API KEY. When provided to E-utilities REST API calls, the max request rate is increased to 10 requests per minute.

See [this page](https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/) for additional documentation.

## Configure a local `.env` file

Once the required and optional external resources have been deployed and configured, it is necessary to configure the Evidence Aggregator to use those resource. This is
done through the creation of a `.env` file at the root of your clone of the EvAgg repository. We have provided a `template.env` file
that can be used as a starting point for generating the `.env` file.

**Note: your `.env` file can potentially contain secrets and should not be committed to source control. The `.env` file is listed in this repository's `.gitignore` file to help prevent this from happening.**

The `.env` file contains a set of environment variables that are used to configure pipeline behavior. To leave any optional settings unset, simply comment them out or delete them from your `.env` file.

Here is an minimal example `.env` file that can be used for a basic test of pipeline functionality if your AOAI Service Resource is configured for EntraID-based authentication/authorization.

```bash
AZURE_OPENAI_DEPLOYMENT="gpt-4-1106-preview" # whatever model name you chose during model deployment
AZURE_OPENAI_ENDPOINT="https://<your_aoai_resource>.openai.azure.com" # https endpoint for AOAI API calls
AZURE_OPENAI_API_VERSION="2024-02-15-preview"
```

The following sections discuss each set of keys in the `template.env` file in more detail.

### OpenAI Service resource configuration

```bash
# Settings for OpenAIClient
AZURE_OPENAI_DEPLOYMENT="<model>" # e.g. gpt-4-1106-preview
AZURE_OPENAI_ENDPOINT="<endpoint>" # https endpoint for AOAI API calls
AZURE_OPENAI_API_KEY="<key>" # [optional] AOAI API key if using key-based authentication.
AZURE_OPENAI_API_VERSION="<version>" # AOAI version (e.g. 2024-02-15-preview)
AZURE_OPENAI_MAX_PARALLEL_REQUESTS=<integer> # [optional] How many simultaneous chat requests to all. (default: 0 [no max])
```

- **AZURE_OPENAI_DEPLOYMENT** - set this to the model deployment name that you configured above. Note that this isn't necessarily the name of the model itself, the model deployment name is customizable during deployment configuration
- **AZURE_OPENAI_ENDPOINT** - the web endpoint for your AOAI Service Resource. This is typically "https://<your_resource_name>.openai.azure.com"
- **AZURE_OPENAI_API_KEY** [optional] - if you are using shared key-based authentication, enter one of your AOAI Service Resource's shared keys here
- **AZURE_OPENAI_API_VERSION** - unless you have a specific reason to do so, set this to "2024-02-15-preview"
- **AZURE_OPENAI_MAX_PARALLEL_REQUESTS** [optional] - this controls the maximum number of concurrent requests being made to the AOAI endpoint. Unless you have a reason to, either don't set this in your `.env` file or set it to 0 (no max).

### NCBI E-utilities configuration [optional]

If you have opted to obtain an API key for the NCBI E-utilities service, the pipeline can be configured to use it with the following keys.

```bash
# Settings for NcbiLookupClient
# If NcbiLookupClient is used and these settings are omitted, NCBI calls are rate-limited to 3rps instead of 10rps.
NCBI_EUTILS_API_KEY="<key>" # [optional] NCBI API key
NCBI_EUTILS_EMAIL="<email>" # [optional] Email address associated with key
```

- **NCBI_EUTILS_API_KEY** [optional] - an API KEY for the NCBI E-utilities web resource
- **NCBI_EUTILS_EMAIL** [optional] - an email address associated with this key

### CosmosDB caching configuration [optional]

If you have opted to deploy a CosmosDB account and database for caching of external API calls and responses, the pipeline can be configured to use this with the following keys.

```bash
# Settings for CosmosCachingWebClient
EVAGG_CONTENT_CACHE_ENDPOINT="<endpoint>" # CosmosDB web endpoint (e.g. https://<your_consmosdb_name>.documents.azure.com:443/)
EVAGG_CONTENT_CACHE_CREDENTIAL="<key>" # [optional] CosmosDB account key if using key-based authentication.
EVAGG_CONTENT_CACHE_CONTAINER="<container>" # [optional] CosmosDB cache container name (default: "cache")
```

- **EVAGG_CONTENT_CACHE_ENDPOINT** - the web endpoint for your CosmosDB instance. This is only necessary if your pipeline execution is leveraging caching.
- **EVAGG_CONTENT_CACHE_CREDENTIAL** [optional] - the account key associated with your CosmosDB. Not needed if you are using EntraID and RBAC-based authentication/authorization.
- **EVAGG_CONTENT_CACHE_CONTAINER** [optional] - the container within your CosmosDB instance that contains the cache records.

## Modify the example pipeline configuration

This repository provides a few configuration files for executing the Evidence Aggregator pipeline in `lib/config/`. Notably, the config `lib/config/evagg_pipeline_example.yaml` is a good starting place for your first full-featured execution of the pipeline. By default, this config assumes the following

- EntraID-based authentication and RBAC-based authorization for use of the AOAI Service resource
- No caching of external web calls and responses
- Processing of a subset of potential papers for two query genes

The following sections discuss modifications that can be made to this `yaml` config to change this default behavior, which may be necessary based on optional steps that may have been completed above.

### Use of shared keys for AOAI authentication/authorization

If are using key-based authentication for the AOAI Service Resource, you will need to comment out or delete the `token_provider` section in each of `lib/config/objects/llm.yaml` and `lib/config/objects/llm_cache.yaml`.

### Use of CosmosDB-based caching of external web calls and responses

Multiple objects described in `lib/config/evagg_pipeline_example.yaml` or their injected dependencies make use of an underlying implementer of `lib.evagg.utils.IWebContentClient`. This codebase provides two implementers of
this interface, one that implements CosmosDB-based caching and one that does not. To switch from using the non-caching web client to the caching web client, all of the instances of the injected web client dependency will need to be changed
to use the caching version. This can be done fairly easily from the top-level config `yaml` by uncommenting the three commented-out lines that terminate in `_cache.yaml` and commenting their counterparts. Note that this config `yaml` already
uses `llm_cache.yaml`. This is unrelated to CosmosDB-based caching and can remain unchanged. Calls to the LLM endpoint are non-deterministic and are thus not cached between runs, but for efficiency, an in-memory cache is used for these calls
during pipeline execution to prevent multiple identical calls.

Note, the above modifications enable use of CosmosDB-based caching using EntraID authentication and role-based authorization. If you are using key-based authentication, you will also need to coment out the `credential` section in `lib/config/objects/web_cache.yaml`.

## Running the pipeline

The script `run_evagg_app` is used to execute a pipeline. It has one required argument - a pointer to a yaml
app spec - and is invoked from the repository root as follows:

```bash
run_evagg_app <path_to_yaml_app_spec_file>
```

The yaml app spec fully defines the behavior of pipeline execution. To perform a basic test of pipeline functionality, after completing the configuration steps above run:

```bash
run_evagg_app lib/config/evagg_pipeline_example.yaml
```

Using gpt-4o-mini with 2000k TPM of quota allocated, this example will complete in approximately 2 minutes. Results of the run will be located
in `.out/run_evagg_pipeline_example_<YYYYMMDD_HHMMSS>`, where each individual run is given a unique datestamp suffix. The primary pipeline output file is `pipeline_benchmark.tsv` contained in that folder.

In a single test run, the pipeline identified 15 unique observations from a total of 4 publications considered in this configuration.

Additional files in this run output folder can be used to debug issues and better understand interactions between the pipeline and the LLM.

You can optionally add or override any leaf value within an app spec dictionary (or sub-dictionary) using
the `-o` argument followed by one or more dictionary `key:value` specifications separated by spaces. The following
command overrides the default log level and outputs the run results to a named file in the `.out` directory:

```bash
run_evagg_app lib/config/evagg_pipeline_example.yaml -o writer.tsv_name:different log.level:DEBUG
```

This will run the same example as above with a different final output filename and increased logging verbosity.
