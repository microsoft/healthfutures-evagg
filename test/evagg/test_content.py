from lib.evagg import Paper, Query, SimpleContentExtractor


def test_simple_content_extractor():
    paper = Paper(
        id="12345678", citation="Doe, J. et al. Test Journal 2021", abstract="This is a test paper.", pmcid="PMC123"
    )
    query = Query(gene="CHI3L1", variant="p.Y34C")
    fields = ["gene", "variant", "MOI", "phenotype", "functional data"]
    extractor = SimpleContentExtractor(fields)
    result = extractor.extract(query=query, paper=paper)
    assert len(result) == 1
    assert result[0]["gene"] == "CHI3L1"
    assert result[0]["variant"] == "p.Y34C"
    assert result[0]["MOI"] == "AD"
    assert result[0]["phenotype"] == "Long face (HP:0000276)"
    assert result[0]["functional data"] == "No"
