from lib.evagg import IExtractFields, IGetPapers, IWriteOutput, Query


class EvAggApp:
    def __init__(self, query: Query, library: IGetPapers, extractor: IExtractFields, writer: IWriteOutput) -> None:
        self._query = query
        self._library = library
        self._extractor = extractor
        self._writer = writer

    def execute(self) -> None:
        # Get the papers that match the query.
        papers = self._library.search(self._query)

        # For all papers that match, extract the fields we want.
        fields = {paper.id: self._extractor.extract(paper) for paper in papers}

        # Write out the result.
        self._writer.write(fields)
