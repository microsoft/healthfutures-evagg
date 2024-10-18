# Evidence Aggregator

The Evidence Aggregator is a collaboration between the
[Broad Institute](https://www.broadinstitute.org/), [Microsoft Research Health Futures](https://www.microsoft.com/en-us/research/lab/microsoft-health-futures/), and the [Centre for Population Genomics](https://populationgenomics.org.au/).
This project aims to support the use of Large Language Models for information retrieval from public literature databases (e.g., PubMed). An example use case is aiding genetic analysts in diagnosis of potential Mendelian disease by searching the the entirety of PubMed for specific details from publications that are potentially relevant to a given case. The code implements a set of Python modules configurable into a pipeline that uses Azure OpenAI to produce output tables with relevant aggregated publication evidence for input gene or genes of interest.

## Quick start

The following steps will allow you to run a test execution of the pipeline that verifies full functionality.

1. [Deploy an AOAI Resource](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal) and [deploy a gpt-4 (or equivalent) model](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal#deploy-a-model)
2. Configure your runtime environment
  a. Install software pre-requisites (`git`, `make`, `jq`, `miniconda`; optionally `az cli` depending on your AOAI authentication approach)
  b. Clone this repository (`git clone https://github.com/microsoft/healthfutures-evagg && cd healthfutures-evagg`)
  c. Build a conda environment (`conda env create -f environment.yml && conda activate evagg`)
  d. Install poetry dependencies (`poetry install`)
  e. Configure `template.env` with your cloud environment settings and save as `.env`
3. Run the pipeline with the example configuration (`run_evagg_app lib/config/evagg_pipeline_example.yaml`)

After pipeline execution is complete, results can be viewed in `.out/run_evagg_pipeline_example_<YYYYMMDD_HHMMSS>`

Each of these steps is described in more detail below.

## Deploying required Azure Resources

The only Azure Resource required to run the Evidence Aggregator pipeline is an Azure OpenAI Service Resource which can be deployed using [these instructions](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal).
Once the AOAI Service Resource is deployed, it is necessary to [deploy a model](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal#deploy-a-model) on that resource that has sufficient capabilities to run the EvAgg pipeline. Specifically, the pipeline currently requires that a model supports at least 128k tokens of input context and the [json output mode](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/json-mode?tabs=python). Currently, the models that meet these two requirements are `gpt-4` (`1106-preview` or newer), `gpt-4o`, and `gpt-4o-mini`. Ensure that one of these models is deployed in your AOAI Service Resource and that you have sufficient quota (at least 100k TPM or equivalent) allocated to that deployment.

Authentication to use the AOAI Service Resource REST APIs can happen either through Role-based access control (RBAC) or shared keys. The EvAgg pipeline currently supports either. After deploying your AOAI Service Resource you will either need to configure RBAC for the identity that will be running the pipeline or take note of one of the shared keys for your AOAI Service Resource. The minimum privilege RBAC role needed to use the AOAI REST APIs during pipeline execution is the `Cognitive Services OpenAI User` role.

In addition to this you will need to make note of the endpoint for AOAI Service Resource (typically `https://<your_resource_name>.openai.azure.com/`) and the name of the model deployment that you configured.

## Configuring your runtime environment

Note, these instructions have been tested on Ubuntu 20.04/22.04 in WSL2 and Azure VMs. Instructions may differ for other platforms.

### Software pre-requisites

- [Python](https://www.python.org/downloads/) 3.12 or above
- azure cli (this is optional depending on whether you intend to authenticate to AOAI via Entra ID/RBAC or using shared keys)

  ```curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash```

- git
- miniconda (installed and configured as below)

    ```bash
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > miniconda.sh
    sh ./miniconda.sh # close and reopen shell.
    /home/azureuser/miniconda3/bin/conda init $SHELL # if you didn't init conda for your shell during setup.
    conda update -n base -c defaults conda -y
    conda config --add channels conda-forge
    conda install -n base conda-libmamba-solver -y
    conda config --set solver libmamba
    ```

If you will be using Entra IDs for authentication to the AOAI Service Resource and any other Azure services, ensure that you have run `az login` and
have a current credential for an identity that is authorized to perform the desired operations.

Additionally, if you plan to perform development tasks for this repo, the following packages are also required:
- make
- jq

### Clone this repository

Create and enter a local clone of this repository in your runtime environment:

```bash
git clone https://github.com/microsoft/healthfutures-evagg
cd healthfutures-evagg
```

### Build a conda environment

Create a conda environment. All shell commands in this section should be executed from the repository's root directory.

```bash
conda env create -f environment.yml
conda activate evagg
```

### Install poetry dependencies

Use poetry to install the local library and register pipeline run application.

```bash
poetry install
```

Test application installation by running the following command:

```bash
run_evagg_app -h
```

You should see a help message displayed providing usage for the `run_evagg_app` application.

### Configure a local `.env` file

In order to communicate with the AOAI Service Resource that you deployed above, the EvAgg codebase will need access to configuration
details that are specific to your cloud environment, **potentially including secrets** like your AOAI Service Resource shared key. This is
done through the creation of a `.env` file at the root of your clone of the EvAgg repository. We have provided a `template.env` file
that can be used as a starting point for generating the `.env` file.

**Note: the `.env` file is listed in this repository's `.gitignore` file to prevent accidentally adding this file to source control.**

The `.env` file contains a set of environment variables that are used to configure pipeline behavior. To leave any optional settings unset, simply comment them out or delete them from your `.env` file.

Here is an minimal example `.env` file that can be used for a basic test of pipeline functionality if your AOAI Service Resource is configured for EntraID-based authentication.

```bash
AZURE_OPENAI_DEPLOYMENT="gpt-4-1106-preview" # whatever model name you chose during model deployment
AZURE_OPENAI_ENDPOINT="https://<your_aoai_resource>.openai.azure.com" # https endpoint for AOAI API calls
AZURE_OPENAI_API_VERSION="2024-02-15-preview"
```

--

The following sections discuss all the variables in `template.env` in more detail.

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

--

The pipeline relies heavily on REST API calls to the NCBI E-utilities service. By default, these requests are throttled to 3 requests per minute. Pipeline execution
can be accelerated by obtaining an API KEY. When provided to E-utilities REST API calls, the max request rate is increased to 10 requests per minute.

```text
# Settings for NcbiLookupClient
# If NcbiLookupClient is used and these settings are omitted, NCBI calls are rate-limited to 3rps instead of 10rps.
NCBI_EUTILS_API_KEY="<key>" # [optional] NCBI API key
NCBI_EUTILS_EMAIL="<email>" # [optional] Email address associated with key
```

- **NCBI_EUTILS_API_KEY** [optional] - an API KEY for the NCBU E-utilities web resource
- **NCBI_EUTILS_EMAIL** [optional] - an email address associated with this key

--

Pipeline execution can be substantially accelerated by creating an Azure-local cache of responses to external web endpoints (e.g., Mutalyzer). Subsequent sections will
discuss how to deploy and configure the Azure resouces associated with this cache, but the following settings are used by the pipeline to identify and communicate with it. They are not necessary if a cache is not being used.

```text
# Settings for CosmosCachingWebClient
EVAGG_CONTENT_CACHE_ENDPOINT="<endpoint>" # CosmosDB web endpoint (e.g. https://<your_consmosdb_name>.documents.azure.com:443/)
EVAGG_CONTENT_CACHE_CREDENTIAL="<key>" # [optional] CosmosDB account key if using key-based authentication.
EVAGG_CONTENT_CACHE_CONTAINER="<container>" # [optional] CosmosDB cache container name (default: "cache")
```

- **EVAGG_CONTENT_CACHE_ENDPOINT** - the web endpoint for your CosmosDB instance. This is only necessary if your pipeline execution is leveraging caching.
- **EVAGG_CONTENT_CACHE_CREDENTIAL** [optional] - the account key associated with your CosmosDB. Not needed if you are using EntraID and RBAC-based authentication/authorization.
- **EVAGG_CONTENT_CACHE_CONTAINER** [optional] - the container within your CosmosDB instance that contains the cache records.

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

Using gpt-4o-mini with 500k TPM of quota allocated (TODO verify), this example will complete in approximately 2 minutes. Results of the run are located
in `.out/run_evagg_pipeline_example_<YYYYMMDD_HHMMSS>`. The primary pipeline output file is `pipeline_benchmark.tsv` contained in that folder.

In a single example run, the pipeline identified 15 unique observations from a total of 4 publications considered in this configuration.

Additional files in this run output folder can be used to debug issues and better understand interactions between the pipeline and the LLM.

You can optionally add or override any leaf value within an app spec dictionary (or sub-dictionary) using
the `-o` argument followed by one or more dictionary `key:value` specifications separated by spaces. The following
command overrides the default log level and outputs the run results to a named file in the `.out` directory:

```bash
run_evagg_app lib/config/evagg_pipeline_example.yaml -o writer.tsv_name:different log.level:DEBUG
```

This will run the same example as above with a different final output filename and increased logging verbosity.

## Deploying and using optional Azure resources

This pipeline makes a substantial number of calls to external web resources, most notably NCBI's E-utilities service and the Mutalyzer service.
These calls can contribute substantially to pipeline execution time as access to those web services is rate limited in accordance with their
usage guidelines. In particular for repeated executions of similarly configured pipelines (e.g., development scenarios), it can substantially
increase performance to enable Azure-local caching of responses from these web services.

The pipeline can be easily configured to use an Azure CosmosDB for this purpose using the following steps.

### Deploy and configure an Azure CosmosDB database

1. Create an Azure CosmosDB account using [these instructions](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/quickstart-portal) TODO: default settings?
2. Within that new account, create a database and container using [these instructions](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/quickstart-portal#create-a-database-and-container). The default container name used by this repository is `cache`, but this can be modified below. TODO: default settings?
3. If using EntraID for authentication configure RBAC settings for your CosmosDB account. A sufficient built-in role to assign is `Cosmos DB Built-in Data Contributor`

### Configure your runtime environment to use this CosmosDB database

Using the following lines from `template.env` as a starting point, modify your `.env` file to contain configuration information for your CosmosDB database.

```bash
# Settings for CosmosCachingWebClient
EVAGG_CONTENT_CACHE_ENDPOINT="<endpoint>" # CosmosDB web endpoint (e.g. https://<your_consmosdb_name>.documents.azure.com:443/)
EVAGG_CONTENT_CACHE_CREDENTIAL="<key>" # [optional] CosmosDB account key if using key-based authentication.
EVAGG_CONTENT_CACHE_CONTAINER="<container>" # [optional] CosmosDB cache container name (default: "cache")
```

See above for more detail on each of these settings. In short, `EVAGG_CONTENT_CACHE_ENDPOINT` is the only required setting unless you are using key-based authentication for CosmosDB, in which case `EVAGG_CONTENT_CACHE_CREDENTIAL` is also required.

### Configure your yaml app spec to enable CosmosDB-based caching

To make use of the CosmosDB database, the pipeline should be configured to use `lib.evagg.utils.CosmosCachingWebClient` in places where classes are making REST API calls (e.g., `lib.evagg.ref.MutalyzerClient`). By default, `lib/config/evagg_pipeline.yaml` is configured to use CosmosDB-based caching
wherever possible and can be used as an example for custom `yaml` app specs.

In short, in the top level `yaml` app spec, or any referenced specs, instances of `di_factory: lib.evagg.utils.RequestsWebContentClient` can be changed to `di_factory: lib.evagg.utils.CosmosCachingWebClient` to enable caching functionality.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for additional detail on guidelines for contribution.

### Code organization

The repository contains the following subdirectories:

`root`  
`|-- data`: sample and reference data  
`|-- lib`: source code for scripts and core libraries  
`|-- scripts`: helper scripts for pre- and post-processing of results  
`|-- test`: `pytest` unit tests for core libraries  
`|-- .out` [generated]: default root directory for pipeline run logging and output  
`|-- .ref` [generated]: default root directory for localized pipeline resources

### Pre-PR checks

Before submitting any PR for review, please verify that linting checks pass (`make lint`) and that tests pass with acceptable coverage for any new code (`make test`). All Pre-PR checks can be run in a single command via `make ci`.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.

## Responsible AI Transparency

### Use of this code

The Evidence Aggregator is intended to be one tool within a genomic analyst's toolkit to review literature related to a variant of interest. It is the user's responsibility to verify the accuracy of the information returned by the Evidence Aggregator. The Evidence Aggregator is for research use only. Users must adhere to the [Microsoft Generative AI Services Code of Conduct](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/code-of-conduct).

The Evidence Aggregator is not designed, intended, or made available for use in the diagnosis, prevention, mitigation, or treatment of a disease or medical condition nor to perform any medical function and the performance of the Evidence Aggregator for such purposes has not been established. You bear sole responsibility for any use of the Evidence Aggregator, including incorporation into any product intended for a medical purpose.

### Limitations

The Evidence Aggregator literature discovery is limited to open-access publications with permissive licenses from the [PubMed Central (PMC) Open Access Subset of journal articles](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/). Information returned by the Evidence aggregator should not be considered exhaustive.

Performance was not optimized for genes with extensive evidence for definitive gene-disease relationships, but for genes with moderate, limited, or no known gene-disease relationship as annotated by ClinGen (Clinical Genome Resource) <https://clinicalgenome.org> [July 2024].

The Evidence Aggregator uses the capabilities of generative AI for both publication foraging and information summarization. Performance of the Evidence Aggregator is limited to the capabilities of the underlying model.  

The design and assessment of the Evidence Aggregator were conducted in English. At present, the Evidence Aggregator is limited to processing inputs and generating outputs in the English language.

## Attributions

National Library of Medicine (US), National Center for Biotechnology Information; [1988]/[cited September 2024]. <https://www.ncbi.nlm.nih.gov/>

Harrison et al., **Ensembl 2024**, Nucleic Acids Research, 2024, 52(D1):D891–D899. PMID: 37953337. <https://doi.org/10.1093/nar/gkad1049>

The Evidence Aggregator uses the Human Phenotype Ontology (HPO version dependent on user environment build and pipeline execution date/time). <http://www.human-phenotype-ontology.org>  

The Evidence Aggregator team thanks the Gene Curation Coalition (GenCC) for providing curated content referenced during development. GenCC’s curated content was obtained at <http://www.thegencc.org> [July 2024] and includes contributions from the following organizations: ClinGen, Ambry Genetics, Franklin by Genoox, G2P, Genomics England PanelApp, Illumina, Invitae, King Faisal Specialist Hospital and Research Center, Laboratory for Molecular Medicine, Myriad Women’s Health, Orphanet, PanelApp Australia.

Environment dependencies may be found in environment.yml.  
