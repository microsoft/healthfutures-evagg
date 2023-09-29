from lib.evagg import Paper, SimpleContentExtractor


def test_simple_content_extractor():
    paper = Paper(id="12345678", citation="Doe, J. et al. Test Journal 2021", abstract="This is a test paper.")
    fields = ["gene", "modification", "MOI", "Phenotype", "Functional data"]
    extractor = SimpleContentExtractor(fields)
    result = extractor.extract(paper)
    assert len(result) == 1
    assert result[0]["gene"] == "CHI3L1"
    assert result[0]["modification"] == "p.Y34C"
    assert result[0]["MOI"] == "AD"
    assert result[0]["Phenotype"] == "Long face (HP:0000276)"
    assert result[0]["Functional data"] == "No"
