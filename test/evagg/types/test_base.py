from lib.evagg.types import HGVSVariant, ICreateVariants, Paper


def test_i_create_variants() -> None:
    class TestCreateVariants(ICreateVariants):
        def try_parse(self, text_desc: str, gene_symbol: str | None, refseq: str | None = None) -> HGVSVariant:
            return HGVSVariant(text_desc, gene_symbol, refseq, False, True)

    test_create_variants = TestCreateVariants()
    variant = test_create_variants.try_parse("var", "gene", "ref")
    assert variant.hgvs_desc == "var"
    assert variant.gene_symbol == "gene"
    assert variant.refseq == "ref"
    assert variant.refseq_predicted == False
    assert variant.valid == True


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
    variant1 = HGVSVariant("var1", "gene1", "ref1", False, True)
    variant2 = HGVSVariant("var1", "gene1", "ref1", False, True)
    variant3 = HGVSVariant("var1", "gene2", "ref2", False, True)

    assert variant1 == variant1
    assert variant1 == variant2
    assert variant1 != variant3
    assert variant1 != "not a variant"


def test_variant_hash() -> None:
    variant1 = HGVSVariant("var1", "gene1", "ref1", False, True)
    variant2 = HGVSVariant("var2", "gene1", "ref1", False, True)
    assert hash(variant1) != hash(variant2)


def test_variant_str() -> None:
    variant = HGVSVariant("var", "gene", "ref", False, True)
    assert str(variant) == "ref:var"


def test_variant_repr() -> None:
    variant = HGVSVariant("var", "gene", "ref", False, True)
    assert variant.__repr__() == "ref:var"
