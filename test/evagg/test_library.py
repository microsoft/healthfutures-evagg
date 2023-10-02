import json
import os
import tempfile
from typing import Sequence

from lib.evagg import Paper, SimpleFileLibrary, Variant


def _paper_to_dict(paper: Paper) -> dict[str, str]:
    return {
        "id": paper.id,
        "citation": paper.citation,
        "abstract": paper.abstract,
    }


def test_search():
    # Create a temporary directory and write some test papers to it
    with tempfile.TemporaryDirectory() as tmpdir:
        paper1 = Paper(id="1", citation="Test Paper 1", abstract="This is a test paper.")
        paper2 = Paper(id="2", citation="Test Paper 2", abstract="This is another test paper.")
        paper3 = Paper(id="3", citation="Test Paper 3", abstract="This is a third test paper.")
        with open(os.path.join(tmpdir, "paper1.json"), "w") as f:
            json.dump(_paper_to_dict(paper1), f)
        with open(os.path.join(tmpdir, "paper2.json"), "w") as f:
            json.dump(_paper_to_dict(paper2), f)
        with open(os.path.join(tmpdir, "paper3.json"), "w") as f:
            json.dump(_paper_to_dict(paper3), f)

        # Create a SimpleFileLibrary instance and search for papers
        library = SimpleFileLibrary(collections=[tmpdir])
        # This should return all papers in the library.
        results = library.search(Variant(gene="test gene", modification="test modification"))

        # Check that the correct papers were returned
        assert len(results) == 3
        print(paper1)
        print(results)

        def _check_for_paper(paper: Paper, results: Sequence[Paper]) -> bool:
            # __eq__ isn't implemented for Paper, and after writing to disk and reloading, the Papers are two different
            # objects.
            if not any(p.id == paper.id for p in results):
                return False
            if not any(p.citation == paper.citation for p in results):
                return False
            if not any(p.abstract == paper.abstract for p in results):
                return False
            return True

        assert _check_for_paper(paper1, results)
        assert _check_for_paper(paper2, results)
        assert _check_for_paper(paper3, results)
