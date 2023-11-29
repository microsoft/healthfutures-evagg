# Sample data

Sample data files are available in this repo for development and testing.

## Data files

- **truth_set_small.tsv**: a collection of 10 genes of interest with a few variants extracted from a small number of papers for each of these genes.
- **truth_set_tiny.tsv**: a subset of the small truth set.
- **truth_set_content_manual.tsv**: a manually curated subset of `truth_set_small.tsv` that contains only papers that are in PMC-OA with a non-ND license and the variants mentioned in the truth set are in the primary text of the paper.

Both of these data files are tab-separated files with the following columns:

- **query_gene**: the query gene which should result in finding this record
- **query_variant**: the query variant which should result in finding this record
- **gene**: the gene impacted by the variant
- **HGVS.C**: the HGVS.C representation of the variant
- **HGVS.P**: the HGVS.P representation of the variant
- **doi**: doi for the paper discussing this variant
- **pmid**: pmid for the paper discussing this variant
- **pmcid**: pmcid for the paper discussing this variant, if one exists
- **is_pmc_oa**: boolean indicator for whether the paper is in the PubMed Central Open Access set, assume false if not provided
- **phenotype**: free text description of observed phenotype for this variant
- **variant_inheritance**: Mendelian inheritance patterns observed for this variant [inherited, de novo, unknown]
- **zygosity**: observed zygosity for this variant [homozygous, heterozygous]
- **study_type**: the type of study [case study, case series, etc ...]
- **functional_study**: description of any functional evidence provided for this variant
- **paper_title**: the title of the paper discussing this variant
- **link**: a link to the paper discussing this variant
- **notes**: any additional notes
