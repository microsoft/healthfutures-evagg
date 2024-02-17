from typing import Protocol

from .base import HGVSVariant


class ICreateVariants(Protocol):
    def try_parse(self, text_desc: str, gene_symbol: str | None, refseq: str | None = None) -> HGVSVariant:
        """Try to parse a variant from a description and gene symbol."""
        ...
