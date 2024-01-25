# %% Imports

import json

from lib.evagg.lit.pubmed import PubtatorEntityAnnotator
from evagg.ref.ncbi import NcbiSnpClient
from lib.evagg.types import Paper
from lib.evagg.web.entrez import BioEntrezClient, BioEntrezDotEnvConfig

# %%


paper = Paper(id="PMC6700409", pmcid="PMC6700409", is_pmc_oa=True)
annotator = PubtatorEntityAnnotator()
result = annotator.annotate(paper)
print(json.dumps(result, indent=2))

# %%

types = set()

for p in result["passages"]:
    for a in p["annotations"]:
        if a["infons"]["type"] == "Mutation":
            types.add(a["infons"]["identifier"])
        #     if a["infons"]["identifier"] == "rs150345688":
        #         po = p['offset']
        #         ao = a['locations'][0]['offset']
        #         print(p['text'][(ao-po-40):(ao-po+40)])

        #         print(a)
# %%


result2 = NcbiSnpClient(BioEntrezClient(BioEntrezDotEnvConfig()))._entrez_fetch_xml("snp", "150345688")
print(result2)

# %%
