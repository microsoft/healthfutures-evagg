# %%
import logging

from lib.evagg.content.observation import make_default_observation_finder
from lib.evagg.types import Paper

logging.basicConfig(level=logging.DEBUG)

paper = Paper(id="pmid:33057194", pmid="33057194", pmcid="PMC7116826", is_pmc_oa=True, license="foo")

finder = make_default_observation_finder()
finder.find_observations(query="BAZ2B", paper=paper)

# %%
