from typing import Dict, List

from lib.evagg.types import IPaperQueryIterator

from ._interfaces import IEvAggApp, IExtractFields, IGetPapers, IWriteOutput


class SynchronousLocalBatchApp(IEvAggApp):
    def __init__(
        self, query_iterator: IPaperQueryIterator, library: IGetPapers, extractor: IExtractFields, writer: IWriteOutput
    ) -> None:
        self._query_factory = query_iterator
        self._library = library
        self._extractor = extractor
        self._writer = writer

    def execute(self) -> None:
        all_fields: Dict[str, List[Dict[str, str]]] = {}

        for query in self._query_factory:
            # Get the papers that match this query.
            papers = self._library.search(query)
            print(f"Found {len(papers)} papers for {query.terms()}")

            for index, paper in enumerate(papers):
                print(f"{index}: {paper.id} - {paper.props['pmcid']}")

            # For all papers that match, extract the fields we want.
            fields = {paper.id: self._extractor.extract(paper, query) for paper in papers}

            # Record the result.
            for paper_id, paper_fields in fields.items():
                if paper_id in all_fields:
                    all_fields[paper_id].extend(paper_fields)
                else:
                    all_fields[paper_id] = list(paper_fields)

        # Write out the result.
        self._writer.write(all_fields)
