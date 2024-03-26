import logging
from typing import Any, Dict, List, Sequence

from .interfaces import IEvAggApp, IExtractFields, IGetPapers, IWriteOutput

logger = logging.getLogger(__name__)


class SynchronousLocalApp(IEvAggApp):
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
        all_fields: Dict[str, List[Dict[str, str]]] = {}

        for query in self._queries:
            if not query.get("gene_symbol"):
                raise ValueError("Minimum requirement to search is to input a gene symbol.")
            term = query["gene_symbol"]
            # Get the papers that match this query.
            papers = self._library.get_papers(query)
            logger.info(f"Found {len(papers)} papers for {term}")

            for index, paper in enumerate(papers):
                logger.debug(f"Paper #{index + 1}: {paper}")

            # For all papers that match, extract the fields we want.
            fields = {paper.id: self._extractor.extract(paper, term) for paper in papers}

            # Record the result.
            for paper_id, paper_fields in fields.items():
                if paper_id in all_fields:
                    all_fields[paper_id].extend(paper_fields)
                else:
                    all_fields[paper_id] = list(paper_fields)

        # Write out the result.
        self._writer.write(all_fields)
