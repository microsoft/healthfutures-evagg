from evagg import IGetPapers, IExtractFields, IWriteOutput

class EvAggApp():
    def __init__(self, query: dict[str, str], library: IGetPapers, extractor: IExtractFields, writer: IWriteOutput) -> None:
        self._query = query
        self._library = library        
        self._extractor = extractor
        self._writer = writer

    def execute(self) -> None:
        # Get the papers that match.
        papers = self._library.search(self._query['gene'], self._query['variant'])

        # For all papers that match, extract the fields we want.
        fields = {paper['id']: self._fields.extract(paper) for paper in papers}

        # Write out the result.
        self._writer.write(fields)