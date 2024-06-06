from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Example:
    gene: str
    text: str


# Each pair of examples is a positive example followed
# by a backup positive example of a different gene.
POSITIVE_EXAMPLES: List[Tuple[Example, Example]] = [
    (
        Example(
            gene="FBN2",
            text="""
Gene: FBN2, PMID: 11285249
Title: Mutation of the gene encoding fibrillin-2 results in syndactyly in mice.
Abstract: Fibrillins are large, cysteine-rich glycoproteins that form microfibrils and play a central role in elastic fibrillogenesis. Fibrillin-1 and fibrillin-2, encoded by FBN1 on chromosome 15q21.1 and FBN2 on chromosome 5q23-q31, are highly similar proteins. The finding of mutations in FBN1 and FBN2 in the autosomal dominant microfibrillopathies Marfan syndrome (MFS) and congenital contractural arachnodactyly (CCA), respectively, has highlighted their essential role in the development and homeostasis of elastic fibres. MFS is characterized by cardiovascular, skeletal and ocular abnormalities, and CCA by long, thin, flexed digits, crumpled ears and mild joint contractures. Although mutations arise throughout FBN1, those clustering within exons 24-32 are associated with the most severe form of MFS, so-called neonatal MFS. All the mutations described in CCA occur in the "neonatal region" of FBN2. Both MFS and CCA are thought to arise via a dominant negative mechanism. The analysis of mouse mutations has demonstrated that fibrillin-1 microfibrils are mainly engaged in tissue homeostasis rather than elastic matrix assembly. In the current investigation, we have analysed the classical mouse mutant shaker-with-syndactylism using a positional candidate approach and demonstrated that loss-of-function mutations outside the "neonatal region" of Fbn2 cause syndactyly in mice. These results suggest that phenotypes distinct from CCA may result in man as a consequence of mutations outside the "neonatal region" of FBN2.
""",
        ),
        Example(
            gene="RHOH",
            text="""
Gene: RHOH, PMID: 22850876
Title: Human RHOH deficiency causes T cell defects and susceptibility to EV-HPV infections.
Abstract: Epidermodysplasia verruciformis (EV) is a rare genetic disorder characterized by increased susceptibility to specific human papillomaviruses, the betapapillomaviruses. These EV-HPVs cause warts and increase the risk of skin carcinomas in otherwise healthy individuals. Inactivating mutations in epidermodysplasia verruciformis 1 (EVER1) or EVER2 have been identified in most, but not all, patients with autosomal recessive EV. We found that 2 young adult siblings presenting with T cell deficiency and various infectious diseases, including persistent EV-HPV infections, were homozygous for a mutation creating a stop codon in the ras homolog gene family member H (RHOH) gene. RHOH encodes an atypical Rho GTPase expressed predominantly in hematopoietic cells. Patients' circulating T cells contained predominantly effector memory T cells, which displayed impaired TCR signaling. Additionally, very few circulating T cells expressed the β7 integrin subunit, which homes T cells to specific tissues. Similarly, Rhoh-null mice exhibited a severe overall T cell defect and abnormally small numbers of circulating β7-positive cells. Expression of the WT, but not of the mutated RHOH, allele in Rhoh-/- hematopoietic stem cells corrected the T cell lymphopenia in mice after bone marrow transplantation. We conclude that RHOH deficiency leads to T cell defects and persistent EV-HPV infections, suggesting that T cells play a role in the pathogenesis of chronic EV-HPV infections.
""",
        ),
    ),
    (
        Example(
            gene="GRXCR2",
            text="""
Gene: GRXCR2, PMID: 24619944
Title: A frameshift mutation in GRXCR2 causes recessively inherited hearing loss.
Abstract: More than 360 million humans are affected with some degree of hearing loss, either early or later in life. A genetic cause for the disorder is present in a majority of the cases. We mapped a locus (DFNB101) for hearing loss in humans to chromosome 5q in a consanguineous Pakistani family. Exome sequencing revealed an insertion mutation in GRXCR2 as the cause of moderate-to-severe and likely progressive hearing loss in the affected individuals of the family. The frameshift mutation is predicted to affect a conserved, cysteine-rich region of GRXCR2, and to result in an abnormal extension of the C-terminus. Functional studies by cell transfections demonstrated that the mutant protein is unstable and mislocalized relative to wild-type GRXCR2, consistent with a loss-of-function mutation. Targeted disruption of Grxcr2 is concurrently reported to cause hearing loss in mice. The structural abnormalities in this animal model suggest a role for GRXCR2 in the development of stereocilia bundles, specialized structures on the apical surface of sensory cells in the cochlea that are critical for sound detection. Our results indicate that GRXCR2 should be considered in differential genetic diagnosis for individuals with early onset, moderate-to-severe and progressive hearing loss.
""",
        ),
        Example(
            gene="MLH3",
            text="""
Gene: MLH3, PMID: 34008015
Title: Novel LRRK2 mutations and other rare, non-BAP1-related candidate tumor predisposition gene variants in high-risk cancer families with mesothelioma and other tumors.
Abstract: There is irrefutable evidence that germline BRCA1-associated protein 1 gene (BAP1) mutations contribute to malignant mesothelioma (MM) susceptibility. However, BAP1 mutations are not found in all cases with evidence of familial MM or in other high-risk cancer families affected by various cancers, including MM. The goal of this study was to use whole genome sequencing (WGS) to determine the frequency and types of germline gene variants occurring in 12 MM patients who were selected from a series of 141 asbestos-exposed MM patients with a family history of cancer but without a germline BAP1 mutation. WGS was also performed on two MM cases, a proband and sibling, from a previously reported family with multiple cases of MM without the inheritance of a predisposing BAP1 mutation. Altogether, germline DNA sequencing variants were identified in 21 cancer-related genes in 10 of the 13 probands. Germline indel, splice site and missense mutations and two large deletions were identified. Among the 13 MM index cases, 6 (46%) exhibited one or more predicted pathogenic mutations. Affected genes encode proteins involved in DNA repair (ATM, ATR, BRCA2, BRIP1, CHEK2, MLH3, MUTYH, POLE, POLE4, POLQ and XRCC1), chromatin modification (ARID1B, DNMT3A, JARID2 and SETD1B) or other cellular pathways: leucine-rich repeat kinase 2 gene (LRRK2) (two cases) and MSH4. Notably, somatic truncating mutation or deletions of LRRK2 were occasionally found in MMs in The Cancer Genome Atlas, and the expression of LRRK2 was undetectable or downregulated in a majority of primary MMs and MM cell lines we examined, implying that loss of LRRK2 expression is a newly recognized tumor suppressor alteration in MM.
""",
        ),
    ),
    (
        Example(
            gene="ADCY1",
            text="""
Gene: ADCY1, PMID: 24482543
Title: Adenylate cyclase 1 (ADCY1) mutations cause recessive hearing impairment in humans and defects in hair cell function and hearing in zebrafish.
Abstract: Cyclic AMP (cAMP) production, which is important for mechanotransduction within the inner ear, is catalyzed by adenylate cyclases (AC). However, knowledge of the role of ACs in hearing is limited. Previously, a novel autosomal recessive non-syndromic hearing impairment locus DFNB44 was mapped to chromosome 7p14.1-q11.22 in a consanguineous family from Pakistan. Through whole-exome sequencing of DNA samples from hearing-impaired family members, a nonsense mutation c.3112C>T (p.Arg1038*) within adenylate cyclase 1 (ADCY1) was identified. This stop-gained mutation segregated with hearing impairment within the family and was not identified in ethnically matched controls or within variant databases. This mutation is predicted to cause the loss of 82 amino acids from the carboxyl tail, including highly conserved residues within the catalytic domain, plus a calmodulin-stimulation defect, both of which are expected to decrease enzymatic efficiency. Individuals who are homozygous for this mutation had symmetric, mild-to-moderate mixed hearing impairment. Zebrafish adcy1b morphants had no FM1-43 dye uptake and lacked startle response, indicating hair cell dysfunction and gross hearing impairment. In the mouse, Adcy1 expression was observed throughout inner ear development and maturation. ADCY1 was localized to the cytoplasm of supporting cells and hair cells of the cochlea and vestibule and also to cochlear hair cell nuclei and stereocilia. Ex vivo studies in COS-7 cells suggest that the carboxyl tail of ADCY1 is essential for localization to actin-based microvilli. These results demonstrate that ADCY1 has an evolutionarily conserved role in hearing and that cAMP signaling is important to hair cell function within the inner ear.
""",
        ),
        Example(
            gene="TOPBP1",
            text="""
Gene: TOPBP1, PMID: 34199176
Title: Novel Genetic and Molecular Pathways in Pulmonary Arterial Hypertension Associated with Connective Tissue Disease.
Abstract: Pulmonary Arterial Hypertension (PAH) is a severe complication of Connective Tissue Disease (CTD), with remarkable morbidity and mortality. However, the molecular and genetic basis of CTD-PAH remains incompletely understood. This study aimed to screen for genetic defects in a cohort of patients with CTD-PAH, using a PAH-specific panel of 35 genes. During recruitment, 79 patients were studied, including 59 Systemic Sclerosis patients (SSc) and 69 females. Disease-associated variants were observed in nine patients: 4 pathogenic/likely pathogenic variants in 4 different genes (TBX4, ABCC8, KCNA5 and GDF2/BMP9) and 5 Variants of Unknown Significance (VUS) in 4 genes (ABCC8, NOTCH3, TOPBP1 and CTCFL). One patient with mixed CTD had a frameshift pathogenic variant in TBX4. Two patients with SSc-PAH carried variants in ABCC8. A patient diagnosed with Systemic Lupus Erythematous (SLE) presented a pathogenic nonsense variant in GDF2/BMP9. Another patient with SSc-PAH presented a pathogenic variant in KCNA5. Four patients with SSc-PAH carried a VUS in NOTCH1, CTCFL, CTCFL and TOPBP1, respectively. These findings suggest that genetic factors may contribute to Pulmonary Vascular Disease (PVD) in CTD patients.
""",
        ),
    ),
    (
        Example(
            gene="SLFN14",
            text="""
Gene: SLFN14, PMID: 30536060
Title: Identification of Two Mutations in PCDHGA4 and SLFN14 Genes in an Atrial Septal Defect Family.
Abstract: Atrial septal defect (ASD) is a common acyanotic congenital cardiac disorder associated with genetic mutations. The objective of this study was to identify the genetic factors in a Chinese family with ASD patients by a whole exome sequencing approach. Causative ASD gene mutations were examined in 16 members from a three-generation family, among which 6 individuals were diagnosed as having ASD. One hundred and eighty-three unrelated healthy Chinese were recruited as a normal control group. Peripheral venous blood was collected from every subject for genetic analysis. Exome sequencing was performed in the ASD patients. Potential causal mutations were detected in non-ASD family members and normal controls by polymerase chain reaction and sequencing analysis. The results showed that all affected family members carried two novel compound mutations, c.1187delT of PCDHGA4 and c.2557insC of SLFN14, and these two mutations were considered to have synergetic function on ASD. In conclusion, the mutations of c.1187delT of PCDHGA4 and c.2557insC of SLFN14 may be pathogenic factors contributing to the development of ASD.
""",
        ),
        Example(
            gene="JPH2",
            text="""
Gene: JPH2, PMID: 23973696
Title: Mutation E169K in junctophilin-2 causes atrial fibrillation due to impaired RyR2 stabilization.
Abstract: This study sought to study the role of junctophilin-2 (JPH2) in atrial fibrillation (AF). JPH2 is believed to have an important role in sarcoplasmic reticulum (SR) Ca(2+) handling and modulation of ryanodine receptor Ca(2+) channels (RyR2). Whereas defective RyR2-mediated Ca(2+) release contributes to the pathogenesis of AF, nothing is known about the potential role of JPH2 in atrial arrhythmias. Screening 203 unrelated hypertrophic cardiomyopathy patients uncovered a novel JPH2 missense mutation (E169K) in 2 patients with juvenile-onset paroxysmal AF (pAF). Pseudoknock-in (PKI) mouse models were generated to determine the molecular defects underlying the development of AF caused by this JPH2 mutation. PKI mice expressing E169K mutant JPH2 exhibited a higher incidence of inducible AF than wild type (WT)-PKI mice, whereas A399S-PKI mice expressing a hypertrophic cardiomyopathy-linked JPH2 mutation not associated with atrial arrhythmias were not significantly different from WT-PKI. E169K-PKI but not A399A-PKI atrial cardiomyocytes showed an increased incidence of abnormal SR Ca(2+) release events. These changes were attributed to reduced binding of E169K-JPH2 to RyR2. Atrial JPH2 levels in WT-JPH2 transgenic, nontransgenic, and JPH2 knockdown mice correlated negatively with the incidence of pacing-induced AF. Ca(2+) spark frequency in atrial myocytes and the open probability of single RyR2 channels from JPH2 knockdown mice was significantly reduced by a small JPH2-mimicking oligopeptide. Moreover, patients with pAF had reduced atrial JPH2 levels per RyR2 channel compared to sinus rhythm patients and an increased frequency of spontaneous Ca(2+) release events. Our data suggest a novel mechanism by which reduced JPH2-mediated stabilization of RyR2 due to loss-of-function mutation or reduced JPH2/RyR2 ratios can promote SR Ca(2+) leak and atrial arrhythmias, representing a potential novel therapeutic target for AF.
""",
        ),
    ),
]


# Each pair of examples is a negative example followed
# by a backup negative example of a different gene.
NEGATIVE_EXAMPLES: List[Tuple[Example, Example]] = [
    (
        Example(
            gene="MLH3",
            text="""
Gene: MLH3, PMID: 38260514
Title: All three MutL complexes are required for repeat expansion in a human stem cell model of CAG-repeat expansion mediated glutaminase deficiency.
Abstract: The Repeat Expansion Diseases (REDs) arise from expansion of a disease-specific short tandem repeat (STR). Different REDs differ with respect to the repeat involved, the cells that are most expansion prone and the extent of expansion and whether these diseases share a common expansion mechanism is unclear. To date, expansion has only been studied in a limited number of REDs. Here we report the first studies of the expansion mechanism in induced pluripotent stem cells derived from a patient with a form of the glutaminase deficiency disorder known as Global Developmental Delay, Progressive Ataxia, And Elevated Glutamine (GDPAG; OMIM# 618412) caused by the expansion of a CAG-STR in the 5' UTR of the glutaminase (GLS) gene. We show that alleles with as few as ~100 repeats show detectable expansions in culture despite relatively low levels of R-looped formed at this locus. Additionally, using a CRISPR-cas9 knockout approach we show that PMS2 and MLH3, the constituents of MutLα and MutLγ, the 2 mammalian MutL complexes known to be involved in mismatch repair (MMR), are essential for expansion. Furthermore, PMS1, a component of a less well understood MutL complex, MutLβ, is also important, if not essential, for repeat expansion in these cells. Our results provide insights into the factors important for expansion and lend weight to the idea that, despite some differences, many, if not all, REDs likely expand via in very similar ways.
""",
        ),
        Example(
            gene="KMO",
            text="""
Gene: KMO, PMID: 33750843
Title: Ablation of kynurenine 3-monooxygenase rescues plasma inflammatory cytokine levels in the R6/2 mouse model of Huntington's disease.
Abstract: Kynurenine 3-monooxygenase (KMO) regulates the levels of neuroactive metabolites in the kynurenine pathway (KP), dysregulation of which is associated with Huntington's disease (HD) pathogenesis. KMO inhibition leads to increased levels of neuroprotective relative to neurotoxic metabolites, and has been found to ameliorate disease-relevant phenotypes in several HD models. Here, we crossed KMO knockout mice to R6/2 HD mice to examine the effect of KMO depletion in the brain and periphery. KP genes were dysregulated in peripheral tissues from R6/2 mice and KMO ablation normalised levels of a subset of these. KP metabolites were also assessed, and KMO depletion led to increased levels of neuroprotective kynurenic acid in brain and periphery, and dramatically reduced neurotoxic 3-hydroxykunurenine levels in striatum and cortex. Notably, the increased levels of pro-inflammatory cytokines TNFa, IL1β, IL4 and IL6 found in R6/2 plasma were normalised upon KMO deletion. Despite these improvements in KP dysregulation and peripheral inflammation, KMO ablation had no effect upon several behavioural phenotypes. Therefore, although genetic inhibition of KMO in R6/2 mice modulates several metabolic and inflammatory parameters, these do not translate to improvements in primary disease indicators-observations which will likely be relevant for other interventions targeted at peripheral inflammation in HD.
""",
        ),
    ),
    (
        Example(
            gene="EMC1",
            text="""
Gene: EMC1, PMID: 36316541
Title: Establishment, characterization and functional testing of two novel ex vivo extraskeletal myxoid chondrosarcoma (EMC) cell models.
Abstract: Extraskeletal myxoid chondrosarcoma (EMC) is a malignant mesenchymal neoplasm of uncertain differentiation as classified by the WHO Classification of Tumours 2020. Although often associated with pronlonged survival, EMC has high rates of distant recurrences and disease-associated death. EMCs are translocation sarcomas and harbor in > 90% of the cases an NR4A3 rearrangement. The molecular consequences of the NR4A3 gene fusions are not yet fully elucidated as well-characterized ex vivo cell models for EMC are lacking. Patient-derived ex vivo models are important and essential tools for investigating disease mechanisms associated with diseases that are rare, that exhibit poor prognosis and for the identification of potential novel treatment options. We established two novel EMC ex vivo models (USZ20-EMC1 and USZ22-EMC2) for functional testing and research purposes. USZ20-EMC1 and USZ22-EMC2 were established and maintained as sarco-sphere cell models for several months in culture. The cells were molecularly characterized using DNA sequencing and methylation profiling. Both cell models represent their native tumor tissue as confirmed by histomorphology and their molecular profiles, suggesting that native tumor cell function can be recapitulated in the ex vivo models. Using a functional screening approach, novel anti-cancer drug sensitivities including potential synergistic combinations were identified. In conclusion, two novel EMC ex vivo cell models (USZ20-EMC1 and USZ22-EMC2) were successfully established and characterized from native tumor tissues. Both cell models will be useful tools for further investigating disease mechanisms and for answering basic and translational research questions.
""",
        ),
        Example(
            gene="FBN2",
            text="""
Gene: FBN2, PMID: 28176809
Title: Unusual life cycle and impact on microfibril assembly of ADAMTS17, a secreted metalloprotease mutated in genetic eye disease.
Abstract: Secreted metalloproteases have diverse roles in the formation, remodeling, and the destruction of extracellular matrix. Recessive mutations in the secreted metalloprotease ADAMTS17 cause ectopia lentis and short stature in humans with Weill-Marchesani-like syndrome and primary open angle glaucoma and ectopia lentis in dogs. Little is known about this protease or its connection to fibrillin microfibrils, whose major component, fibrillin-1, is genetically associated with ectopia lentis and alterations in height. Fibrillin microfibrils form the ocular zonule and are present in the drainage apparatus of the eye. We show that recombinant ADAMTS17 has unique characteristics and an unusual life cycle. It undergoes rapid autocatalytic processing in trans after its secretion from cells. Secretion of ADAMTS17 requires O-fucosylation and its autocatalytic activity does not depend on propeptide processing by furin. ADAMTS17 binds recombinant fibrillin-2 but not fibrillin-1 and does not cleave either. It colocalizes to fibrillin-1 containing microfibrils in cultured fibroblasts and suppresses fibrillin-2 (FBN2) incorporation in microfibrils, in part by transcriptional downregulation of Fbn2 mRNA expression. RNA in situ hybridization detected Adamts17 expression in specific structures in the eye, skeleton and other organs, where it may regulate the fibrillin isoform composition of microfibrils.
""",
        ),
    ),
    (
        Example(
            gene="TAPBP",
            text="""
Gene: TAPBP, PMID: 24159917
Title: Gastrointestinal stromal tumors: a case-only analysis of single nucleotide polymorphisms and somatic mutations.
Abstract: Gastrointestinal stromal tumors are rare soft tissue sarcomas that typically develop from mesenchymal cells with acquired gain-in-function mutations in KIT or PDGFRA oncogenes. These somatic mutations have been well-characterized, but little is known about inherited genetic risk factors. Given evidence that certain susceptibility loci and carcinogens are associated with characteristic mutations in other cancers, we hypothesized that these signature KIT or PDGFRA mutations may be similarly fundamental to understanding gastrointestinal stromal tumor etiology. Therefore, we examined associations between 522 single nucleotide polymorphisms and seven KIT or PDGFRA tumor mutations types. Candidate pathways included dioxin response, toxin metabolism, matrix metalloproteinase production, and immune and inflammatory response. We estimated odds ratios and 95% confidence intervals for associations between each candidate SNP and tumor mutation type in 279 individuals from a clinical trial of adjuvant imatinib mesylate. We used sequence kernel association tests to look for pathway-level associations. One variant, rs1716 on ITGAE, was significantly associated with KIT exon 11 non-codon 557-8 deletions (odds ratio = 2.86, 95% confidence interval: 1.71-4.78) after adjustment for multiple comparisons. Other noteworthy associations included rs3024498 (IL10) and rs1050783 (F13A1) with PDGFRA mutations, rs2071888 (TAPBP) with wild type tumors and several matrix metalloproteinase SNPs with KIT exon 11 codon 557-558 deletions. Several pathways were strongly associated with somatic mutations in PDGFRA, including defense response (p = 0.005) and negative regulation of immune response (p = 0.01). This exploratory analysis offers novel insights into gastrointestinal stromal tumor etiology and provides a starting point for future studies of genetic and environmental risk factors for the disease.
""",
        ),
        Example(
            gene="GRXCR2",
            text="""
Gene: GRXCR2, PMID: 34366792
Title: Murine GRXCR1 Has a Different Function Than GRXCR2 in the Morphogenesis of Stereocilia.
Abstract: Mutations in human glutaredoxin domain-containing cysteine-rich protein 1 (GRXCR1) and its paralog GRXCR2 have been linked to hearing loss in humans. Although both GRXCR1 and GRXCR2 are required for the morphogenesis of stereocilia in cochlear hair cells, a fundamental question that remains unclear is whether GRXCR1 and GRXCR2 have similar functions in hair cells. Previously, we found that GRXCR2 is critical for the stereocilia morphogenesis by regulating taperin localization at the base of stereocilia. Reducing taperin expression level rescues the morphological defects of stereocilia and hearing loss in Grxcr2-deficient mice. So far, functions of GRXCR1 in mammalian hair cells are still unclear. Grxcr1-deficient hair cells have very thin stereocilia with less F-actin content inside, which is different from Grxcr2-deficient hair cells. In contrast to GRXCR2, which is concentrated at the base of stereocilia, GRXCR1 is diffusely distributed throughout the stereocilia. Notably, GRXCR1 interacts with GRXCR2. In Grxcr1-deficient hair cells, the expression level of GRXCR2 and taperin is reduced. Remarkably, different from that in Grxcr2-deficient mice, reducing taperin expression level does not rescue the morphological defects of stereocilia or hearing loss in Grxcr1-deficient mice. Thus, our findings suggest that GRXCR1 has different functions than GRXCR2 during the morphogenesis of stereocilia.
""",
        ),
    ),
    (
        Example(
            gene="TNNC2",
            text="""
Gene: TNNC2, PMID: 34502093
Title: Troponin Variants in Congenital Myopathies: How They Affect Skeletal Muscle Mechanics.
Abstract: The troponin complex is a key regulator of muscle contraction. Multiple variants in skeletal troponin encoding genes result in congenital myopathies. TNNC2 has been implicated in a novel congenital myopathy, TNNI2 and TNNT3 in distal arthrogryposis (DA), and TNNT1 and TNNT3 in nemaline myopathy (NEM). Variants in skeletal troponin encoding genes compromise sarcomere function, e.g., by altering the Ca2+ sensitivity of force or by inducing atrophy. Several potential therapeutic strategies are available to counter the effects of variants, such as troponin activators, introduction of wild-type protein through AAV gene therapy, and myosin modulation to improve muscle contraction. The mechanisms underlying the pathophysiological effects of the variants in skeletal troponin encoding genes are incompletely understood. Furthermore, limited knowledge is available on the structure of skeletal troponin. This review focusses on the physiology of slow and fast skeletal troponin and the pathophysiology of reported variants in skeletal troponin encoding genes. A better understanding of the pathophysiological effects of these variants, together with enhanced knowledge regarding the structure of slow and fast skeletal troponin, will direct the development of treatment strategies.
""",
        ),
        Example(
            gene="OTUD7A",
            text="""
Gene: OTUD7A, PMID: 26110020
Title: Partial tetrasomy of the proximal long arm of chromosome 15 in two patients: the significance of the gene dosage in terms of phenotype.
Abstract: Large amounts of low copy number repeats in the 15q11.2q13.3 chromosomal region increase the possibility of misalignments and unequal crossover during meiosis in this region, leading to deletions, duplications, triplications and supernumerary chromosomes. Most of the reported cases with epilepsy, autism and Prader-Willi/Angelman syndrome are in association with rearrangements of the proximal long arm of chromosome 15. Here we report the first two unrelated Hungarian patients with the same epileptic and dysmorphic features, who were investigated by array comparative genomic hybridization (array CGH). By G-banded karyotype followed by FISH and array CGH we could detect partial tetrasomy of the 15q11.2q13.3 chromosomal region, supporting proximal 15q duplication syndrome. Findings of the array CGH gave fully explanation of the phenotypic features of these patients, including epileptic seizures, delayed development, hyperactivity and craniofacial dysmorphic signs. Besides the described features of isodicentric (15) (idic(15)) syndrome Patient 1. suffered from bigeminic extrasystoles and had postnatal growth retardation, which had been published only in a few articles. Dosage effect of some genes in the concerned genomic region is known, but several genes have no evidence to have dosage dependence. Our results expanded the previous literature data. We assume dosage dependence in the case of CHRNA7 and OTUD7A, which might be involved in growth regulation. On the other hand increased dosage of the KLF13 gene seems to have no direct causal relationship with heart morphology. The genomic environment of the affected genes may be responsible for the observed phenotype.
""",
        ),
    ),
]
