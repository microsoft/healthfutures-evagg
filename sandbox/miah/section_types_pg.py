# %% Imports.

from lib.di import DiContainer
from lib.evagg.content.fulltext import get_sections

# %% Build some instances.

container = DiContainer()
ts = container._instance_from_yaml("lib/config/objects/truthset.yaml", parameters={}, resources={})
queries = container._instance_from_yaml("lib/config/queries/content_subset.yaml", parameters={}, resources={})

# %%

all_types = set()

for q in queries:
    papers = ts.get_papers(query=q)
    for p in papers:
        if p.props.get("can_access", False):
            sections = get_sections(p.props["fulltext_xml"])
            print(these_types := {s.section_type for s in sections})
            all_types.update(these_types)

print(all_types)
# %%
