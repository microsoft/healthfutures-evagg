"""sk_content_pg.py.

Playground for experimentation on content extraction using SemanticKernel.
"""

# %% Imports.
import json
from typing import Any, List, Sequence

import requests
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureTextCompletion

from lib.config import PydanticYamlModel
from lib.evagg.sk import SemanticKernelClient, SemanticKernelDotEnvConfig

# %% Constants.
pmid = "33739604"
query_gene = "PRKCG"
query_variant = None
pmcid = "PMC8045942"

# %% Handle config.

config = SemanticKernelDotEnvConfig()

# # %% Load in, pre-process a paper.

# # Get the paper from pubmed
# # from Bio import Entrez
# # Entrez.email = "miah@microsoft.com"

# # handle = Entrez.efetch(db="pubmed", id=PMID, retmode="xml")
# # records = Entrez.read(handle)
# # handle.close()

# """Fetch a paper from PubMed Central using the BioC API."""
# r = requests.get(f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmid}/ascii", timeout=10)
# r.raise_for_status()
# d = r.json()

# %%
# if len(d["documents"]) != 1:
#     raise ValueError(f"Expected a single document, got {len(d['documents'])}")

# doc = d["documents"][0]

# %% exploring doc['passages']
# This is where the meat of the document is.

# each passage has an "infons" dict, which houses the relevant metadata
#   has "section_type"
#   has "type"
#
# each passage has a "text" field, which is the actual text of the passage

# set([p['infons']['section_type'] for p in doc['passages']])
# set([p['infons']['type'] for p in doc['passages']])

# [len(p["text"]) for p in doc["passages"]]

# %% Use pubtator API for entity recognition.

format = "json"
r = requests.get(
    f"https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/bioc{format}?pmcids={pmcid}", timeout=10
)
r.raise_for_status()
d = r.json()

# This gives us a fat pile of annotations of extracted entities, but it only gives you the official symbol for each
# gene, so we have to know that the query gene is the official one. This info can be pulled from NCBI's gene database.

# %% Ok, let's try to narrow down to passages that only contain the named entity.
for p in d["passages"]:
    for a in p["annotations"]:
        if a["text"] == query_gene:
            print(f"{query_gene} found in passage {p['infons']['section']}:")
            for loc in a["locations"]:
                o = loc["offset"] - p["offset"]
                print(f"  {p['text'][max(o-200, 0):(o+200)]}")

# %% Yeah, that works, but better yet, I think we can just use the variants (which have an RSID) and then go ask which
# of those RSIDs are on the query gene.


# No, even better, the annotation includes the Gene ID.


def _gene_symbol_for_id(ids: Sequence[str]) -> dict[str, str]:
    # TODO, wrap in Bio.Entrez library as they're better about rate limiting and such.
    url = f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha/gene/id/{','.join(ids)}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return {g["gene"]["gene_id"]: g["gene"]["symbol"] for g in r.json()["reports"]}


def _gene_id_for_symbol(symbols: Sequence[str]) -> dict[str, str]:
    # TODO, wrap in Bio.Entrez library as they're better about rate limiting and such.

    url = f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha/gene/symbol/{','.join(symbols)}/taxon/Human"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return {g["gene"]["symbol"]: g["gene"]["gene_id"] for g in r.json()["reports"]}


# lookup = _gene_symbol_for_id(["4508", "12345"])
# print(lookup)

# foo = _gene_id_for_symbol(["COQ2", "KRAS"])
# print(foo)

# TODO, random idea for later, point back to clinvar records for variants mentioned in papers

# %% Now that we can get gene ids, get the id for the query gene.

lookup = _gene_id_for_symbol([query_gene])
query_gene_id = int(lookup[query_gene])

# %% Then search all of the Mutation entities for the query gene id.

variants_in_query_gene: List[dict[str, Any]] = []

for p in d["passages"]:
    for a in p["annotations"]:
        if a["infons"]["type"] == "Mutation":
            if "gene_id" in a["infons"] and int(a["infons"]["gene_id"]) == query_gene_id:
                variants_in_query_gene.append(a)

# %%


def _dedup_variants(variants_in: Sequence[dict[str, Any]]) -> Sequence[dict[str, Any]]:
    identifiers = {v["infons"]["identifier"] for v in variants_in}
    variants = []
    for identifier in identifiers:
        matches = [v for v in variants_in if v["infons"]["identifier"] == identifier]
        entity_ids = {v["id"] for v in matches}
        variant = matches[0]
        variant["entity_ids"] = entity_ids
        variants.append(variant)
    return variants


deduped = _dedup_variants(variants_in_query_gene)

# %% Now we've got a gene, and a list of variants within the document that are on that gene. Let's see what content
# we can extract about them.

# TODO: There are better ways to save this above
# get passages associated with each variant

contexts = {}

for idx in range(len(deduped)):
    txt = ""

    for p in d["passages"]:
        in_passage = any(
            (a["infons"]["type"] == "Mutation") and (a["id"] in deduped[idx]["entity_ids"]) for a in p["annotations"]
        )

        if in_passage:
            print(f"variant {[deduped[idx]['text']]} found in passage:")
            print(f"  {p['text']}")
            txt = txt + "\n" + p["text"] if txt else p["text"]

    variables = {"variant": deduped[idx]["text"], "gene": query_gene}
    context_variables = sk.ContextVariables(content=txt, variables=variables)
    contexts[deduped[idx]["text"]] = context_variables

# %% Extract content using evagg library code.

client = SemanticKernelClient(config)

for variant, vars in contexts.items():
    print(f"Processing variant: {variant}")
    variables_dict = vars.dict()["variables"]
    print(client.run_completion_function("content", "phenotype", variables_dict))


# %% Extract content.

# For the potentially relevant chunks, extract candidate field values.

kernel = sk.Kernel()

kernel.add_text_completion_service("dv", AzureTextCompletion(config.deployment, config.endpoint, config.api_key))

prompt = """
# Task
Given a text from a paper, please output the reported pattern of inheritance for a variant/gene pair. 
Please output: de novo, dominant, recessive, X-linked or not specified. 

# Output format
Report only one of the following: de novo, dominant, recessive, X-linked or not specified. 
Additionally, provide grounding from the input text that supports the reported MOI.

The the entirety of your response should be a json blob of the following format:
```
{
  "moi": "de novo" | "dominant" | "recessive" | "X-linked" | "not specified",
  "grounding": "evidence in support of the moi."
}
```

# Definitions
De novo - The variant is present in the proband only and not parents 
Dominant - Each affected person has an affected parent; occurs in every generation
Recessive - Both parents of an affected person are carriers; not typically seen in every generation 
X-linked - associated with mutations in genes on the X chromosome, can be donominant or recessive
not specified - information about de novo, dominant, recessive, or X-linked are not found in the paper

#Examples:
Input: Table 1
Molecular data of individuals with SRSF1 variants

Subjects	Genomic changeg	Coding changeh	Protein change	Variant type	Inheritance	Polyphen-2 score	Varsome predictiona,b,c,d,e,f
I1	g.56083708_56083709del	c.377_378del	p.Ser126Trpfs∗17	Frameshift	Not inherited from the mother	–	Pathogenic: PVS1 PM2 PS2
I2, I3, I4	g.56083236C>T	c.478G>A	p.Val160Met	Missense	De novo	Probably damaging 0.999	Likely pathogenic: PS2 PM2 PP3
I5	g.56082937dup	c.579dup	p.Val194Serfs∗2	Frameshift	De novo	–	Pathogenic: PVS1 PM2 PS2
I6	g.55806534_56540597del	Not applicable	Not applicable	Deletion	De novo	–	–
I7	g.56084417G>A	c.82C>T	p.Arg28∗	Nonsense	De novo	–	Pathogenic: PVS1 PM2 PS2
I8	g.56083166T>C	c.548A>G	p.His183Arg	Missense	De novo	Probably damaging 1	Likely pathogenic: PS2 PM2 PP3
I9	g.56084380C>A	c.119G>T	p.Gly40Val	Missense	De novo	Probably damaging 1	Likely pathogenic: PS2 PM2 PP3
I10	g.56084402C>A	c.97G>T	p.Glu33∗	Nonsense	De novo	–	Pathogenic: PVS1 PM2 PS2
I11	g.56082914del	c.601del	p.Ser201Valfs∗87	Frameshift	De novo	–	Pathogenic: PVS1 PM2 PS2
I12	g.56083875C>T	c.208G>A	p.Ala70Thr	Missense	De novo	Probably damaging 1	Likely pathogenic: PS2 PM2 PP3
I13	g.56084428G>A	c.71C>T	p.Pro24Leu	Missense	De novo	Probably damaging 1	Likely pathogenic: PP3 PM2 PP2 PS2
I14	g.56083852A>C	c.231T>G	p.Tyr77∗	Nonsense	De novo	–	Pathogenic: PVS1 PM2 PS2
I15	g.56083832A>C	c.251T>G	p.Leu84Arg	Missense	De novo	Probably damaging 1	Likely pathogenic: PS2 PM2 PP3
I16	g.56084369C>T	c.130G>A	p.Asp44Asn	Missense	De novo	Benign 0.029	Uncertain significance: PP2 PM2 PS2 BP4
I17	g.55442363_56309063del	Not applicable	Not applicable	Deletion	De novo	–	–
Variant: c.119G>T; p.Gly40Val
Gene: SRSF1
Output: 
{
  "moi": "de novo",
  "grounding": "Subject I9 has the variant in question and it it was reported as a de novo mutation."
}

---

Input: Cystic fibrosis (CF; OMIM #219700) is the most frequent autosomal recessive inherited genetic disease of the 
Caucasian race. Its incidence is 1/2500-3500 newborns.1,2 It is more frequent in European and European-origin 
communities when compared with other communities.3 Recently, we showed the incidence of CF to be 1/3400 live births in 
our region, similar to other European communities.4 While the percentage of carriers differs according to ethnic 
origins, it is 1/25 in Northern Europe, 1/29 among Ashkenazi Jews,5,6 1/46 in Hispanic Americans, and 1/65 in African 
Americans.7

The cystic fibrosis transmembrane conductance regulatory (CFTR) gene contains 27 exons and is localized on 7q31. The 
protein is composed of 1480 amino acids. CFTR is an integral membrane protein that operates as a regulated chloride 
channel in the epithelia. Cystic fibrosis occurs due to homozygous or compound heterozygous pathogenic variants in the 
CFTR (CFTR/ABCC7: MIM*602421) gene. Today, more than 2000 disease-causing variants of the CFTR gene have been found and
their ethnic and geographical distributions show differences.8 The most common variant seen in almost all the studies is
F508del. In addition, the frequency of more than 20 variants is more significant than 0.1% 
(www.genet.sickkid.on.ca/Home.html.).
Variant: c.865_869delAGACA; p.R289Nfs*17
Gene: CFTR
Output: 
{
  "moi": "recessive",
  "grounding": "Cystic fibrosis is reported as an autosomal recessive disease."
}

---

Input: {{$input}}
Variant: {{$variant}}
Gene: {{$gene}}
Output: 
"""

func = kernel.create_semantic_function(prompt, max_tokens=2000, temperature=0.2)

for variant, vars in contexts.items():
    print(f"Processing variant: {variant}")
    summary = func.invoke(variables=vars)
    print(summary)

# # If needed, async is available too: summary = await summarize.invoke_async(input_text)
# summary = func(variables=context_variables)

# print(summary)

# Now structure the result as a list of dicts, where the keys in each dict correspond to the fields, and each element
# in the list corresponds to a gene/variant pair.

# %%

prompt2 = """
Here is some text from a paper describing genetic variants in the gene {{$gene}}.
What is the described mode of inheritance for patients with the variant: {{$variant}}?

{{$input}}

Provide your response in the following format:
{
  "grounding": "evidence in support of the moi.",
  "moi": "de novo" | "dominant" | "recessive" | "X-linked" | "not specified"
}

"""

func2 = kernel.create_semantic_function(prompt2, max_tokens=100, temperature=0.2)

for variant, vars in contexts.items():
    print(f"Processing variant: {variant}")
    print(func2.invoke(variables=vars))

# %%
