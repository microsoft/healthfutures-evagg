import logging
from typing import Any, Dict, List, Sequence

from lib.evagg.utils.run import set_run_complete

from .interfaces import IEvAggApp, IExtractFields, IGetPapers, IWriteOutput

logger = logging.getLogger(__name__)


class PaperQueryApp(IEvAggApp):
    def __init__(
        self,
        queries: Sequence[Dict[str, Any]],
        library: IGetPapers,
        extractor: IExtractFields,
        writer: IWriteOutput,
    ) -> None:
        self._queries = queries
        self._library = library
        self._extractor = extractor
        self._writer = writer

    def execute(self) -> None:
        output_fieldsets: List[Dict[str, str]] = []

        for query in self._queries:
            if not query.get("gene_symbol"):
                raise ValueError("Minimum requirement to search is to input a gene symbol.")
            term = query["gene_symbol"]
            # Get the papers that match this query.
            papers = self._library.get_papers(query)
            # Assert each returned paper has a unique id
            assert len(papers) == len({p.id for p in papers})
            logger.info(f"Found {len(papers)} papers for {term}")

            # Extract observation fieldsets for each paper.
            for paper in papers:
                extracted_fieldsets = self._extractor.extract(paper, term)
                output_fieldsets.extend(extracted_fieldsets)

        # Write out the results.
        output_file = self._writer.write(output_fieldsets)
        set_run_complete(output_file)
