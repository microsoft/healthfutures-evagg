# Evidence Aggregator

The Evidence Aggregator is a collaboration between the [Broad Institute](https://www.broadinstitute.org/), [Microsoft Research Health Futures](https://www.microsoft.com/en-us/research/lab/microsoft-health-futures/), and the [Centre for Population Genomics](https://populationgenomics.org.au/).

This project leverages Generative AI to drastically accelerate targeted information retrieval and reasoning over large text knowledge bases (e.g. PubMed). We focus here on rare disease diagnostics, where retrieval of relevant, current information from the scientific literature is particularly challenging. The Evidence Aggregator is an analytic Copilot that can query the entirety of PubMed, highlighting relevant literature for a given gene of interest, detailing information including described genetic variants, the patients they impact, and the phenotypic consequences. This approach is readily scalable to other applications and the modularity of the codebase readily supports this.

## Using Evidence Aggregator

Evidence Aggregator functionality is exposed as a library of independent pipeline components, implemented as Python functions or classes, for querying, filtering, and aggregating genetic variant publication evidence. These components are designed to be assembled in various ways into a pipeline "app" that can be run to generate some sort of concrete analysis or output - for example, to take a set of gene names as input and output a table of PubMed publication data referencing variants in those genes. The repo contains a number of existing pipeline definitions that can be run as-is or modified/reconfigured to produce different results.

### Pipeline apps

An "app" is any Python class that implements the `lib.evagg.IEvAggApp` protocol (effectively an `execute` method) to be instantiated and run via the `run_evagg_app` Linux command-line entrypoint. An app is defined in a yaml specification file as an ordered dictionary of key/value pairs describing the full class/parameter component hierarchy to be instantiated before `execute` is called. The abstract format of a yaml spec is as follows:

```yaml
# Reusable resource definitions.
resource_1: <value or spec sub-dictionary>
---
resource_n: <value or spec sub-dictionary>

# Application definition.
di_factory: <fully-qualified class name, factory method, or yaml file path>
param_1: <value or spec sub-dictionary>
---
param_n: <value or spec sub-dictionary>
```

The `lib.di` module is responsible for parsing, instantiating, and returning an arbitrary app object from an app spec when `run_evagg_app` is called. For a given spec, it first collects, in file order, the top-level name/value pairs - "resources" are entries occurring before the `di_factory` key, and "parameters" are those occurring after. It then resolves any string value of the form `"{{resource_name}}"` by looking `resource_name` up in the "resources" collection - this allows instantiation of singleton objects earlier in the spec that may be reused at multiple places later in the spec. Finally, the `di` module instantiates and returns the top-level app object represented by the spec by invoking the `di_factory:` entrypoint value, passing in the resolved/collected parameters as named keyword arguments. If any resource or parameter value in the spec consists of a sub-dictionary with its own `di_factory` key, that value is first resolved to a runtime object, recursively, following the same mechanism just described for the top-level spec. Object hierarchies of this sort may be arbitrarily deep.

The existing app/sub-object spec files for defining various runnable pipelines are found in `lib/config/`.

### Quickstart

The following setup steps will allow you to run a simple pipeline app at the Linux command-line that outputs (fabricated) sample results without relying on any external resources. (Each step is described in greater detail for increased pipeline functionality in [SETUP.md](SETUP.md).)

1. [Install software prerequisites](SETUP.md#install-software-prerequisites): `python`, `git`, `miniconda/libmamba`
2. [Clone this repository](SETUP.md#clone-the-repository): `git clone https://github.com/microsoft/healthfutures-evagg && cd healthfutures-evagg`
3. [Build a conda environment](SETUP.md#build-a-conda-environment): `conda env create -f environment.yml && conda activate evagg`
4. [Install poetry dependencies](SETUP.md#install-poetry-dependencies): `poetry install`

Then run the sample pipeline using the following command. It will output a few lines of placeholder publication "evidence" to standard output.

```bash
run_evagg_app lib/config/sample_config.yaml
```

Proceed to [SETUP.md](SETUP.md) to set up external dependencies and perform a full-featured example execution of the pipeline against live resources.

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

Before submitting any PR for review, please verify that linting checks pass (`make lint`) and that tests pass with acceptable coverage for any new code (`make test`). All pre-PR checks can be run in a single command via `make ci`.

### PMC Cache Setup (Optional)

For improved performance and offline access, you can set up a local cache of PMC (PubMed Central) archives:

1. **Create cache directory:**

   ```bash
   mkdir -p .cache/pmc
   cd .cache/pmc
   ```

2. **Download PMC archives using lftp:**

   ```bash
   lftp ftp.ncbi.nlm.nih.gov:/pub/wilbur/BioC-PMC -e "mget -c -P 5 PMC*xml_ascii.tar.gz; quit"
   ```

   **Note:** The PMC mirror is updated 3-4 times per year. To keep your cache current, set up a crontab to refresh periodically:

   ```bash
   # Add to crontab (crontab -e) to check for updates monthly
   0 2 1 * * cd /path/to/.cache/pmc && lftp ftp.ncbi.nlm.nih.gov:/pub/wilbur/BioC-PMC -e "mget -c -P 5 PMC*xml_ascii.tar.gz; quit"
   ```

   The `-c` flag ensures efficient mirroring by only downloading new/updated files.

3. **Create index files (first time setup):**

   ```bash
   uv run python -c "import glob; from ratarmountcore.mountsource.factory import open_mount_source; [open_mount_source(f) for f in glob.glob('.cache/pmc/PMC*_xml_ascii.tar.gz')]"
   ```

   **Note:** This indexing step can take a while the first time but significantly improves subsequent access performance.

4. **Enable mirror in pipeline configuration:**
   Edit your pipeline YAML config (e.g., `lib/config/objects/ncbi.yaml`) and uncomment the `pmc_mirror_dir` parameter:
   ```yaml
   pmc_mirror_dir: ".cache/pmc" # Uncomment and set path to enable local PMC mirror
   ```

When enabled, the system will first attempt to fetch papers from the local mirror before falling back to online retrieval.

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
