# Sample data

Sample data files are available in this repo for development and testing.

## Data files (v1)

The v1 data files represent the EvAgg v1 train and test datasets for both paper finding and content extraction.

- **v1/group_assignments.tsv**: Assignment of the v1 genes to train or test groups.
Generated using sandbox/ash/parse_paper_finding_ground_truth.ipynb.
- **v1/papers_[train, test]_v1.tsv**: paper-gene pairings of the complete set of papers that should be found for a given
gene.
- **v1/evidence_[train, test]_v1.tsv**: row-by-row extracted evidence for the [train, test] sets. These are both tab
separated files with the following columns:
  - **gene**: the gene of interest
  - **paper_build**: the reference genome build referred to in the paper, if provided
  - **paper_variant**: the text description of the variant provided in the paper
  - **individual_id**: an identifier of the individual possessing the variant if provided
  - **pheno_text_description**: a free text description of the individual's phenotype as provided in the paper
  - **phenotype**: an HPO/OMIM translation of the free text phenotype description
  - **transcript**: a RefSeq or Ensembl identifier of the transcript of interest, if provided
  - **hgvs_c**: a correctly-formatted hgvs.c representation of the variant description, if available
  - **hgvs_p**: a correctly-formatted hgvs.p representation of the variant description, if applicable and available
  - **variant_type**: the type of variant
  - **zygosity**: the observed zygosity in the individual, if available
  - **variant_inheritance**: the pattern of inheritance in the individual, if available
  - **pmid**: the pmid for the paper of interest
  - **author**: the last name of the author of the paper of interest
  - **year**: the year of publication of the paper of interest
  - **journal**: the journal in which the paper of interest was published
  - **study_type**: the type of study
  - **engineered_cells**: boolean indicator of whether functional study was carried out using engineered cells
  - **patient_cells_tissues**: boolean indicator of whether functional study was carried out using patient cells/tissues
  - **animal_model**: boolean indicator of whether functional study was carried out in an animal model
  - **in_supplement**: boolean indicator of whether the first mention of the variant was in the supplement
  - **notes**: any additional notes from the curator
  - **questions**: any additional questions from the curator
  - **paper_id**: pmid-prefixed paper id
  - **paper_title**: the title of the paper
  - **can_access**: boolean indicator of whether the paper permits full-text access
  - **license**: indicator of the license terms, if available
  - **pmcid**: pubmed central id for the paper, if available
  - **link**: link to the paper on pubmed
  - **evidence_base**: gencc evidence base for the gene
  - **group**: train/test group assignment

## Archival data files

- **truth_set_small.tsv**: a collection of 10 genes of interest with a few variants extracted from a small number of papers for each of these genes.
- **truth_set_tiny.tsv**: a subset of the small truth set.
- **truth_set_content_manual.tsv**: a manually curated subset of `truth_set_small.tsv` that contains only papers that are in PMC-OA with a non-ND license and the variants mentioned in the truth set are in the primary text of the paper.

Both of these data files are tab-separated files with the following columns:

- **query_gene**: the query gene which should result in finding this record
- **query_variant**: the query variant which should result in finding this record
- **gene**: the gene impacted by the variant
- **hgvs_c**: the HGVS.C representation of the variant
- **hgvs_p**: the HGVS.P representation of the variant
- **doi**: doi for the paper discussing this variant
- **pmid**: pmid for the paper discussing this variant
- **pmcid**: pmcid for the paper discussing this variant, if one exists
- **is_pmc_oa**: boolean indicator for whether the paper is in the PubMed Central Open Access set, assume false if not provided
- **license**: if available, the license type for the paper
- **phenotype**: free text description of observed phenotype for this variant
- **variant_inheritance**: Mendelian inheritance patterns observed for this variant [inherited, de novo, unknown]
- **zygosity**: observed zygosity for this variant [homozygous, heterozygous]
- **study_type**: the type of study [case study, case series, etc ...]
- **functional_study**: description of any functional evidence provided for this variant
- **paper_title**: the title of the paper discussing this variant
- **link**: a link to the paper discussing this variant
- **notes**: any additional notes
