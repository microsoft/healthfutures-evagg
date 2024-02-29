from typing import Any, Dict, Optional, Protocol, Sequence

from lib.evagg.types import Paper


class IVariantLookupClient(Protocol):
    def hgvs_from_rsid(self, *rsids: str) -> Dict[str, Dict[str, str]]:
        """Get HGVS variants for the given rsids."""
        ...  # pragma: no cover


class IGeneLookupClient(Protocol):
    def gene_id_for_symbol(self, *symbols: str, allow_synonyms: bool = False) -> Dict[str, int]:
        """Get gene ids for the given gene symbols."""
        ...  # pragma: no cover


class IPaperLookupClient(Protocol):
    def search(self, query: str, max_papers: Optional[int] = None) -> Sequence[str]:
        """Search the paper database for the given query."""
        ...  # pragma: no cover

    def fetch(self, paper_id: str) -> Optional[Paper]:
        """Fetch the paper with the given id."""
        ...  # pragma: no cover

    def full_text(self, paper: Paper, kept_section_types: Optional[Sequence[str]] = None) -> Optional[str]:
        """Fetch the full text of the paper."""
        ...


class IAnnotateEntities(Protocol):
    def annotate(self, paper: Paper) -> Dict[str, Any]:
        """Annotate entities in the paper."""
        ...  # pragma: no cover


class IRefSeqLookupClient(Protocol):
    def transcript_accession_for_symbol(self, symbol: str) -> str | None:
        """Get 'Reference Standard' RefSeq accession ID for the given gene symbol."""
        ...  # pragma: no cover

    def protein_accession_for_symbol(self, symbol: str) -> str | None:
        """Get 'Reference Standard' RefSeq protein accession ID for the given gene symbol."""
        ...  # pragma: no cover


class INormalizeVariants(Protocol):
    def normalize(self, hgvs: str) -> Dict[str, Any]:
        """Perform normalization on the provided variant."""
        ...  # pragma: no cover


class IBackTranslateVariants(Protocol):
    def back_translate(self, hgvsp: str) -> Sequence[str]:
        """Back translate the provided protein variant.

        Returns all possible coding transcript variants that could give rise to the provided protein variant.
        """
        ...  # pragma: no cover
