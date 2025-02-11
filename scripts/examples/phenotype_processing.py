"""This script provides examples of how to configure portions of the EvAgg pipeline to process phenotypes->HPO."""

# %% Imports.

import asyncio

import nest_asyncio

from lib.di import DiContainer
from lib.evagg.content import PromptBasedContentExtractor

# %% Asyncio setup, this is necessary because we want to run an asyncio method in a notebook context.

# More on this, so it's not just mystery. Because EvAgg makes a lot of calls out to the internet when it's running,
# we used an asynchronous development model. The very short (and possibly wrong) description of this is that when
# calling a method from within the system, we don't actually block execution of the program waiting for the response
# unless we have to. This is great from a software engineering perspective because it really speeds things up, but it
# sucks from a learning perspective because it makes the program a little more confusing to use. Anything in this
# example that you see that has "asyncio" in the name is a result of this. Further, I wanted to give you examples that
# would both run interactively (in a Jupyter Notebook) but also as a command line script, so there's even more goop
# here to make that work.
nest_asyncio.apply()

# %% Build objects.

# Here I'm going to cheat and actually instantiate all of EvAgg's object stack.
app = DiContainer().create_instance({"di_factory": "lib/config/evagg_pipeline_example.yaml"}, {})

# This does two nice things for us
# 1. It gives us a fully configured instance of the PromptBasedContentExtractor
# 2. It uses all the existing logging machinery, so we get a record of all the prompts and responses.

# To achieve (1), we can just reach in and grab it.
extractor: PromptBasedContentExtractor = app._extractor

# To achieve (2), once the first line from this cell has been run, take a look at .out/run_kernel...
# This folder will contain a set of *.log files that correspond to all of the LLM calls. This can be used for
# subsequent debugging.

# %% Find the phenotypes in a patient description.

# This is a synthetic patient description with association with a specific real individual.
free_text = """
The patient is a 7-year-old male diagnosed with Dravet Syndrome, a rare and severe form of epilepsy. His medical 
history reveals that his first seizure occurred at 6 months old, triggered by a high fever. Since then, he has 
experienced frequent and prolonged seizures, both febrile and afebrile. Genetic testing confirmed a mutation in the 
SCN1A gene, which is commonly associated with Dravet Syndrome. Despite early developmental milestones being met, the 
patient has shown signs of developmental delays, particularly in speech and motor skills.

In addition to seizures, the patient exhibits some of the classical phenotypes of Dravet Syndrome, such as 
hyperactivity and cognitive impairment. However, he does not display autistic-like behaviors, which are often seen in 
other patients with this condition. His seizures are often triggered by hyperthermia, such as during physical 
activity or illness, and he has a history of status epilepticus, requiring emergency medical intervention on multiple 
occasions. The patient is currently on multiple anti-seizure medications, but his seizures remain refractory to 
treatment.

The patient's overall health is closely monitored due to the risk of sudden unexpected death in epilepsy (SUDEP) and 
other complications associated with Dravet Syndrome. He attends a specialized school where he receives support for his 
cognitive and motor impairments. His family is actively involved in his care, ensuring he avoids known seizure triggers 
and adheres to his medication regimen. Despite the challenges, the patient maintains a positive outlook and engages in 
various activities with appropriate supervision.
"""

# This function will take a block of text and return a list of the phenotypes found in it that are associated with the
# patient described in the "description" argument, and the gene sybmol provided in the "metadata" argument.

# These constraints aren't ideal for all applications, since this was designed for the EvAgg use case, but this can
# still provide a reasonable starting point.
phenotype_strings = asyncio.run(
    extractor._observation_phenotypes_for_text(text=free_text, description="patient", metadata={"gene_symbol": "SCN1A"})
)

print(phenotype_strings)

# This should run three prompts and return a list of strings that represent the phenotypes found in the text.
# e.g.,
# [
# 'Dravet Syndrome',
# 'seizures',
# 'developmental delays',
# 'speech delays',
# 'motor skills delays',
# 'hyperactivity',
# 'cognitive impairment',
# 'status epilepticus'
# ]

# Further, if you look in the corresponding output folder, you can see log files for those three prompts, and these log
# files will contain the "transcripts" of the prompts and responses to the LLM.

# %% Convert phenotype strings to HPO terms.

# This function will take that list of string phenotypes and attempt to convert them to HPO terms.
# Again, because this was implemented for EvAgg use, it's not perfect for all applications (e.g., you don't get to know
# which input string matches to which HPO term), but it's a good starting point.
hpo_terms = asyncio.run(extractor._convert_phenotype_to_hpo(phenotype_strings.copy()))

# This will run potentially a bunch of different prompts and the output will look something like...
# [
# 'Delayed speech and language development (HP:0000750)',
# 'Dravet Syndrome',
# 'Status epilepticus (HP:0002133)',
# 'Cognitive impairment (HP:0100543)',
# 'Global developmental delay (HP:0001263)',
# 'Motor delay (HP:0001270)',
# 'Hyperactivity (HP:0000752)',
# 'Seizure (HP:0001250)'
# ]

# Note that Dravet Syndrome isn't an HPO term, so the system gave up and just returned the raw string.

print(hpo_terms)
