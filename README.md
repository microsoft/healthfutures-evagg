# Evidence Aggregator

The Evidence Aggregator is a collaboration between the [Broad Institute](https://www.broadinstitute.org/), [Microsoft Research Health Futures](https://www.microsoft.com/en-us/research/lab/microsoft-health-futures/), and the [Centre for Population Genomics](https://populationgenomics.org.au/).

This project leverages Generative AI to drastically accelerate targeted information retrieval and reasoning over large text knowledge bases (e.g. PubMed). We focus here on rare disease diagnostics, where retrival of relevant, current information from the scientific literature is particularly challenging. The Evidence Aggregator is an evidence aggregation Copilot that can query the entirety of PubMed, highlighting relevant literature for a given gene of interest, detailing information including described genetic variants, the patients they impact, and the phenotypic consequences. This approach is readily scalable to other applications and the modularity of the codebase readily supports this.

## Running Evidence Aggregator

The following section walks you through how to configure your software environment to successfully run the Evidence Aggregator. After successfully completing and testing your environment setup, you can proceed to [Resource Configuration](RESOURCE_CONFIG.md) to set up external dependencies and perform a full-featured test execution of the pipeline.

### Overview - standalone example

The following steps will allow you to run a test execution of the pipeline that verifies software pre-requisites.

1. [Configure your runtime environment](#configuring-your-runtime-environment)
    1. Install software pre-requisites (`git`, `make`, `jq`, `miniconda`, `az cli`)
    2. Clone this repository (`git clone https://github.com/microsoft/healthfutures-evagg && cd healthfutures-evagg`)
    3. Build a conda environment (`conda env create -f environment.yml && conda activate evagg`)
    4. Install poetry dependencies (`poetry install`)
2. [Run the pipeline](#running-the-pipeline---standalone-example) with the example configuration (`run_evagg_app lib/config/sample_config.yaml`).

### Configuring your runtime environment

Note, these instructions have been tested on Ubuntu 20.04/22.04 in WSL2 and Azure VMs. Instructions may differ for other platforms.

### Software pre-requisites

- [Python](https://www.python.org/downloads/) 3.12 or above
- azure cli (this is optional depending on whether you intend to authenticate to AOAI via Entra ID or using shared keys)

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

### Running the pipeline - standalone example

Now that your software pre-requisites are configured correctly, run the following command to execute the pipeline using a sample configuration. Note when using this configuration, the Evidence Aggregator will not try to leverage external dependencies (e.g., AOAI Service Resource, NCBI's E-utilities, Mutalyzer, etc). Correspondingly, the outputs of the pipeline for this example
are not particularly interesting. However, this does permit you to validate whether your software pre-requisites are correctly installed and configured.

Run the pipeline using the following command:

```bash
run_evagg_app lib/config/sample_config.yaml
```

The pipeline should output a few lines of placeholder "evidence" to standard output.

Proceed to [Resource Configuration](RESOURCE_CONFIG.md) to set up external dependencies and perform a full-featured test execution of the pipeline.

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

Performance was not optimized for genes with extensive evidence for definitive gene-disease relationships, but for genes with moderate, limited, or no known gene-disease relationship as annotated in Gene Curation Coalition (GenCC) <http://www.thegencc.org> [July 2024].

The Evidence Aggregator uses the capabilities of generative AI for both publication foraging and information summarization. Performance of the Evidence Aggregator is limited to the capabilities of the underlying model.  

The design and assessment of the Evidence Aggregator were conducted in English. At present, the Evidence Aggregator is limited to processing inputs and generating outputs in the English language.

## Attributions

National Library of Medicine (US), National Center for Biotechnology Information; [1988]/[cited September 2024]. <https://www.ncbi.nlm.nih.gov/>

Harrison et al., **Ensembl 2024**, Nucleic Acids Research, 2024, 52(D1):D891–D899. PMID: 37953337. <https://doi.org/10.1093/nar/gkad1049>

Lefter M et al. (2021). Mutalyzer 2: Next Generation HGVS Nomenclature Checker. Bioinformatics, 2021 Sep 15; 37(28):2811-7

The Evidence Aggregator uses the Human Phenotype Ontology (HPO version dependent on user environment build and pipeline execution date/time). <http://www.human-phenotype-ontology.org>  

The Evidence Aggregator team thanks the Gene Curation Coalition (GenCC) for providing curated content referenced during development. GenCC’s curated content was obtained at <http://www.thegencc.org> [July 2024] and includes contributions from the following organizations: ClinGen, Ambry Genetics, Franklin by Genoox, G2P, Genomics England PanelApp, Illumina, Invitae, King Faisal Specialist Hospital and Research Center, Laboratory for Molecular Medicine, Myriad Women’s Health, Orphanet, PanelApp Australia.

Environment dependencies may be found in environment.yml.  
