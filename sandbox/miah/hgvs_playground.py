"""This notebook is for playing around with the hgvs package."""

# %% Imports.

from lib.evagg.lit.pubmed import PubtatorEntityAnnotator
from lib.evagg.types import Paper

# %% First, let's generate a candidate set of variants that we'll want to parse.

PMCIDS = ["PMC6912785", "PMC6175136"]

papers = [Paper(**{"id": pmcid, "is_pmc_oa": True, "pmcid": pmcid}) for pmcid in PMCIDS]
# Use pubtator to get variants for these PMIDs.

PubtatorEntityAnnotator().annotate
