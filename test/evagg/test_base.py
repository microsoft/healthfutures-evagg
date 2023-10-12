from lib.evagg import Paper


def test_repr():
    # Test case 1: abstract length is less than 10
    obj = Paper(id="1", citation="", abstract="hello", pmcid="PMC123")
    assert repr(obj) == 'id: 1 - abstract: "hello"'

    # Test case 2: abstract length is greater than 10
    obj = Paper(id="2", citation="", abstract="hello world!", pmcid="PMC123")
    assert repr(obj) == 'id: 2 - abstract: "hello worl..."'


def test_paper_from_dict():
    paper_dict = {"id": "123", "citation": "Test Citation", "abstract": "Test Abstract", "pmcid": "PMC123"}
    paper = Paper.from_dict(paper_dict)
    assert paper.id == "123"
    assert paper.citation == "Test Citation"
    assert paper.abstract == "Test Abstract"
    assert paper.pmcid == "PMC123"
