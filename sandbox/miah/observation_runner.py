# %%

import asyncio

from lib.di import DiContainer
from lib.evagg.app import PaperQueryApp

app: PaperQueryApp = DiContainer().create_instance(spec={"di_factory": "lib/config/evagg_pipeline.yaml"}, resources={})

# %%

paper = app._library._paper_client.fetch("25885783", include_fulltext=True)
result = asyncio.run(app._extractor._observation_finder.find_observations("FAAH2", paper))

for r in result:
    print(
        f"{r.individual} / {r.variant} ({'+' if r.variant.valid else '-'}): {r.variant_descriptions} - {r.patient_descriptions}"
    )
