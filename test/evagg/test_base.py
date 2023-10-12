from lib.evagg import Paper


def test_paper_from_dict():
    paper_dict = {"id": "123", "citation": "Test Citation", "abstract": "Test Abstract", "pmcid": "PMC123"}
    paper = Paper(**paper_dict)
    assert paper.id == "123"
    assert paper.citation == "Test Citation"
    assert paper.abstract == "Test Abstract"
    assert paper.props["pmcid"] == "PMC123"
