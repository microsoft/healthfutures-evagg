from ._interfaces import IEvAggApp, IExtractFields, IGetPapers, IPaperQuery, IWriteOutput


class SynchronousLocalApp(IEvAggApp):
    def __init__(
        self, query: IPaperQuery, library: IGetPapers, extractor: IExtractFields, writer: IWriteOutput
    ) -> None:
        self._query = query
        self._library = library
        self._extractor = extractor
        self._writer = writer

    def execute(self) -> None:
        # Get the papers that match the query.
        papers = self._library.search(self._query)
        print(f"Found {len(papers)} papers for {self._query.terms()}")

        # For all papers that match, extract the fields we want.
        fields = {paper.id: self._extractor.extract(paper, self._query) for paper in papers}

        # Write out the result.
        self._writer.write(fields)
