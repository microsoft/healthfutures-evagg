# %%

from lib.evagg.content import HGVSVariantComparator, HGVSVariantFactory, ObservationFinder
from lib.evagg.llm import OpenAIClient
from lib.evagg.ref import MutalyzerClient, NcbiLookupClient, NcbiReferenceLookupClient
from lib.evagg.svc import RequestsWebContentClient, get_dotenv_settings
from lib.evagg.types import HGVSVariant

# %%

# web_client:
#   di_factory: lib.evagg.svc.CosmosCachingWebClient
#   cache_settings:
#     di_factory: lib.evagg.svc.get_dotenv_settings
#     filter_prefix: "EVAGG_CONTENT_CACHE_"

# ncbi_client:
#   di_factory: lib.evagg.ref.NcbiLookupClient
#   web_client: "{{web_client}}"
#   settings:
#     di_factory: lib.evagg.svc.get_dotenv_settings
#     filter_prefix: "NCBI_EUTILS_"

# mutalyzer_client:
#   di_factory: lib.evagg.ref.MutalyzerClient
#   web_client:
#     di_factory: lib.evagg.svc.CosmosCachingWebClient
#     web_settings:
#       # Mutalyzer response code of 422 signifies an unprocessable entity.
#       no_raise_codes: [422]
#     cache_settings:
#       di_factory: lib.evagg.svc.get_dotenv_settings
#       filter_prefix: "EVAGG_CONTENT_CACHE_"

# refseq_client:
#   di_factory: lib.evagg.ref.NcbiReferenceLookupClient
#   web_client:
#     di_factory: lib.evagg.svc.RequestsWebContentClient

# variant_factory:
#   di_factory: lib.evagg.content.HGVSVariantFactory
#   validator: "{{mutalyzer_client}}"
#   normalizer: "{{mutalyzer_client}}"
#   variant_lookup_client: "{{ncbi_client}}"
#   refseq_client: "{{refseq_client}}"


ncbi_client = NcbiLookupClient(
    web_client=RequestsWebContentClient(), settings=get_dotenv_settings(filter_prefix="NCBI_EUTILS_")
)

mutalyzer_client = MutalyzerClient(web_client=RequestsWebContentClient(settings={"no_raise_codes": [422]}))

refseq_client = NcbiReferenceLookupClient(web_client=RequestsWebContentClient())
variant_factory = HGVSVariantFactory(
    validator=mutalyzer_client,
    normalizer=mutalyzer_client,
    variant_lookup_client=ncbi_client,
    refseq_client=refseq_client,
)

# variant_comparator:
#   di_factory: lib.evagg.content.HGVSVariantComparator

variant_comparator = HGVSVariantComparator()

#   observation_finder:
#     di_factory: lib.evagg.content.ObservationFinder
#     llm_client: "{{llm_client}}"
#     paper_lookup_client: "{{ncbi_client}}"
#     variant_factory: "{{variant_factory}}"
#     variant_comparator:
#       di_factory: lib.evagg.content.HGVSVariantComparator
#     normalizer: "{{mutalyzer_client}}"

llm_client = OpenAIClient(get_dotenv_settings(filter_prefix="AZURE_OPENAI_"))

observation_finder = ObservationFinder(llm_client, ncbi_client, variant_factory, variant_comparator, mutalyzer_client)

# %% Make some variants to compare.

# variant_descriptions = ["c.3344A>T (NM_001999.4)", "c.3344A>T", "p.D1115V"]
# gene_symbol = "FBN2"
# genome_build = None

# variants_by_description = {
#     description: variant
#     for description in variant_descriptions
#     if (variant := observation_finder._create_variant_from_text(description, gene_symbol, genome_build)) is not None
# }

v1 = variant_factory.parse("c.1309C>T", "EXOC2", "NM_018303.6")
v2 = variant_factory.parse("p.Arg437*", "EXOC2", "NM_018303.6(NP_060773.3)")

consolidated = variant_comparator.consolidate([v1, v2], True)

# %%
