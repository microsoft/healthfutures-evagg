from typing import Any, Dict, Optional, Protocol, Sequence, Tuple

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
    def search(self, query: str, **extra_params: Dict[str, Any]) -> Sequence[str]:
        """Search the paper database for the given query."""
        ...  # pragma: no cover

    def fetch(self, paper_id: str, include_fulltext: bool = False) -> Optional[Paper]:
        """Fetch the paper with the given id."""
        ...  # pragma: no cover


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

    def genomic_accession_for_symbol(self, symbol: str) -> str | None:
        """Get 'Reference Standard' RefSeq genomic accession ID for the given gene symbol."""
        ...

    def accession_autocomplete(self, accession: str) -> Optional[str]:
        """Get the latest RefSeq version for a versionless accession."""
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


class IValidateVariants(Protocol):
    def validate(self, hgvs: str) -> bool:
        """Validate the provided variant."""
        ...  # pragma: no cover


class ICompareHPO(Protocol):
    def compare(self, subject: str, object: str, method: str) -> float:
        """Compare two HPO terms using the specified method.

        HPO terms should be provided as strings, e.g. "HP:0012469"
        """
        ...  # pragma: no cover

    def compare_set(self, subjects: Sequence[str], objects: Sequence[str], method: str) -> Dict[str, Tuple[float, str]]:
        """Compare two sets of HPO terms using the specified method.

        HPO terms should be provided as a sequence of strings, e.g. ["HP:0012469", "HP:0007270"]
        Will return a dictionary mapping each subject term to a tuple containing the maximum similarity score from
        objects, and the term in objects corresponding to that score.
        """
        ...  # pragma: no cover


class ITranslateTextToHPO(Protocol):
    def translate(self, text: str) -> str:
        """Translate a text description to an HPO term.

        Returns the HPO term as a string, e.g. "HP:0012469"
        """
        ...  # pragma: no cover
