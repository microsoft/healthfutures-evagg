import logging
from typing import Dict, List

from lib.evagg.types import IPaperQueryIterator

from .interfaces import IEvAggApp, IExtractFields, IGetPapers, IWriteOutput

logger = logging.getLogger(__name__)


class SynchronousLocalApp(IEvAggApp):
    def __init__(
        self,
        queries: IPaperQueryIterator,
        library: IGetPapers,
        extractor: IExtractFields,
        writer: IWriteOutput,
    ) -> None:
        self._query_factory = queries
        self._library = library
        self._extractor = extractor
        self._writer = writer

    def execute(self) -> None:
        all_fields: Dict[str, List[Dict[str, str]]] = {}

        for query in self._query_factory:
            # Get the papers that match this query.
            papers = self._library.search(query)
            logger.info(f"Found {len(papers)} papers for {query.terms()}")

            for index, paper in enumerate(papers):
                logger.debug(f"{index}: {paper.id} - {paper.props['pmcid']}")

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
