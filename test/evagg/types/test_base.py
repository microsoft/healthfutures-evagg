from lib.evagg.types import HGVSVariant, ICreateVariants, Paper


def test_i_create_variants() -> None:
    class TestCreateVariants(ICreateVariants):
        def parse_rsid(self, rsid: str) -> HGVSVariant:
            return HGVSVariant("c.123A>C", None, None, False, True, None, [])

        def parse(
            self,
            text_desc: str,
            gene_symbol: str | None,
            refseq: str | None = None,
            protein_consequence: HGVSVariant | None = None,
        ) -> HGVSVariant:
            return HGVSVariant(text_desc, gene_symbol, refseq, False, True, protein_consequence, [])

    test_create_variants = TestCreateVariants()
    variant = test_create_variants.parse("var", "gene", "ref")
    assert variant.hgvs_desc == "var"
    assert variant.gene_symbol == "gene"
    assert variant.refseq == "ref"
    assert variant.refseq_predicted is False
    assert variant.valid is True
    assert variant.protein_consequence is None
    assert variant.coding_equivalents == []


def test_paper_from_dict() -> None:
    paper_dict = {"id": "123", "citation": "Test Citation", "abstract": "Test Abstract", "pmcid": "PMC123"}
    paper = Paper(**paper_dict)
    assert paper.id == "123"
    assert paper.citation == "Test Citation"
    assert paper.abstract == "Test Abstract"
    assert paper.props["pmcid"] == "PMC123"


def test_paper_equality() -> None:
    paper_dict = {"id": "123", "citation": "Test Citation", "abstract": "Test Abstract", "pmcid": "PMC123"}
    same_paper_dict = paper_dict.copy()
    different_paper_dict = paper_dict.copy()
    different_paper_dict["id"] = "456"

    paper = Paper(**paper_dict)
    same_paper = Paper(**same_paper_dict)
    different_paper = Paper(**different_paper_dict)

    assert paper == paper
    assert paper == same_paper
    assert paper != different_paper
    assert paper != "not a paper"


def test_paper_repr() -> None:
    paper_dict = {"id": "123", "citation": "Test Citation", "abstract": "Test Abstract", "pmcid": "PMC123"}
    str_paper = 'id: 123 - "Test Citation"'

    paper = Paper(**paper_dict)
    assert str(paper) == str_paper


def test_variant_equality() -> None:
    variant1 = HGVSVariant("var1", "gene1", "ref1", False, True, None, [])
    variant2 = HGVSVariant("var1", "gene1", "ref1", False, True, None, [])
    variant3 = HGVSVariant("var1", "gene2", "ref2", False, True, None, [])

    assert variant1 == variant1
    assert variant1 == variant2
    assert variant1 != variant3
    assert variant1 != "not a variant"


def test_variant_hash() -> None:
    variant1 = HGVSVariant("var1", "gene1", "ref1", False, True, None, [])
    variant2 = HGVSVariant("var2", "gene1", "ref1", False, True, None, [])
    assert hash(variant1) != hash(variant2)


def test_variant_str() -> None:
    variant = HGVSVariant("var", "gene", "ref", False, True, None, [])
    assert str(variant) == "ref:var"


def test_variant_repr() -> None:
    variant = HGVSVariant("var", "gene", "ref", False, True, None, [])
    assert variant.__repr__() == "ref:var"
