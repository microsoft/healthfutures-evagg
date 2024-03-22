from typing import Protocol

from .base import HGVSVariant


class ICreateVariants(Protocol):
    def parse_rsid(self, rsid: str) -> HGVSVariant:
        """Try to parse a variant from an rsid."""
        ...  # pragma: no cover

    def parse(self, text_desc: str, gene_symbol: str | None, refseq: str | None = None) -> HGVSVariant:
        """Try to parse a variant from a description and gene symbol."""
        ...  # pragma: no cover
