# %%

from lib.di import DiContainer

# %% First problem to explore, how to create and compare genomic and coding variants.

spec = {
    "di_factory": "lib.evagg.content.ObservationFinder",
    "llm_client": {"di_factory": "lib/config/objects/llm.yaml"},
    "variant_factory": {"di_factory": "lib/config/objects/variant_factory.yaml"},
    "variant_comparator": {"di_factory": "lib.evagg.content.HGVSVariantComparator"},
    "normalizer": {"di_factory": "lib/config/objects/mutalyzer.yaml"},
}

of = DiContainer().create_instance(spec=spec, resources={})

# %% Here's an example to mess with.

# (Pdb) variant_str
# '19564510C>T'python
# (Pdb) gene_symbol
# 'EMC1'
# (Pdb) refseq
# 'hg19(chr1)'

# c.1212+1G>A

gvar = of._variant_factory.parse("19564510C>T", "EMC1", "hg19(chr1)")
cvar = of._variant_factory.parse("c.1212+1G>A", "EMC1", "NG_032948.1(NM_015047.2)")

print(of._variant_comparator.compare(gvar, cvar))

# %%

# pmid:32639540	EXOC2	c.1309C>T	p.Arg437Ter	Patient 1:IV-2	NM_018303.6	True
# pmid:32639540	EXOC2	g.576766G>A	NA	Patient 1:IV-2	NC_000006.11	True

gvar2 = of._variant_factory.parse("g.576766G>A", "EXOC2", "NC_000006.11")
cvar2 = of._variant_factory.parse("c.1309C>T", "EXOC2", "NM_018303.6")
pvar2 = of._variant_factory.parse("p.Arg437Ter", "EXOC2", "NP_060773.3")

print(of._variant_comparator.compare(gvar2, cvar2))
print(of._variant_comparator.compare(gvar2, pvar2))
print(of._variant_comparator.compare(cvar2, pvar2))

# %%
