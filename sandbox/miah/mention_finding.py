# %%

from lib.evagg.lit import VariantMentionFinder
from lib.evagg.lit.pubmed import PubtatorEntityAnnotator
from lib.evagg.ref import NcbiGeneClient
from lib.evagg.types import Paper, Query
from lib.evagg.web.entrez import BioEntrezClient, BioEntrezDotEnvConfig

# %%
paper = Paper(id="PMC10277729", pmcid="PMC10277729", is_pmc_oa=True)
query = Query("FAM111B:Thr625Asn")
mention_finder = VariantMentionFinder(
    PubtatorEntityAnnotator(), NcbiGeneClient(BioEntrezClient(BioEntrezDotEnvConfig()))
)

mentions = mention_finder.find_mentions(query, paper)

# %%
mentions.keys()
