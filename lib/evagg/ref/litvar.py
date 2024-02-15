import json
import re
from typing import Any, Mapping, Sequence

import requests
from ratelimit import limits, sleep_and_retry


class LitVarReference:
    # Regular expression for variants with RSIDs, e.g., litvar@rs1002421360##
    LITVAR_RSID_RE = re.compile(r"^litvar%40rs\d+%23%23$")
    # Regular expression for variants with HGVS only, litvar@#673#p.Y1796_1797ins
    LITVAR_HGVS_RE = re.compile(r"^litvar%40%23\d+%23[cpng]\.[A-Za-z0-9\-\*\_\>]+$")
    # Regular expression for ClinGen + RSID variants, e.g., litvar@CA175337#rs397507479##
    LITVAR_CLINGEN_RSID_RE = re.compile(r"^litvar%40CA\d+%23rs\d+%23%23$")

    @classmethod
    def _requests_get(cls, url: str, timeout: int = 10) -> requests.Response:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r

    @classmethod
    def _requests_get_text(cls, url: str, timeout: int = 10) -> str:
        return cls._requests_get(url, timeout=timeout).text

    @classmethod
    @sleep_and_retry
    @limits(calls=3, period=1)
    # This expression actually gets to the RateLimitDecorator class instance that implements the
    # rate limiting: lv._requests_limited.__closure__[0].cell_contents.__closure__[1].cell_contents
    # In theory could use this change the rate limiting parameters at runtime.
    def _requests_get_text_limited(cls, url: str, timeout: int = 10) -> str:
        return cls._requests_get_text(url, timeout=timeout)

    @classmethod
    def variant_autocomplete(cls, query: str, limit: int = 100) -> Sequence[Mapping[str, Any]]:
        # Example usage
        # https://www.ncbi.nlm.nih.gov/research/litvar2-api/variant/autocomplete/?query=p.V600E
        if not query or len(query) == 0:
            return []

        url = f"https://www.ncbi.nlm.nih.gov/research/litvar2-api/variant/autocomplete/?query={query}&limit={limit}"
        response_text: str = cls._requests_get_text_limited(url)  # type: ignore
        return json.loads(response_text)

    # Note that this will return variants of a few different naming conventions...
    # They represent different objects in the DB even if they're biologically the same variant.

    @classmethod
    def variants_for_gene(cls, gene_symbol: str) -> Sequence[Mapping[str, Any]]:
        # Example usage
        # https://www.ncbi.nlm.nih.gov/research/litvar2-api/variant/search/gene/BRAF
        if not gene_symbol or len(gene_symbol) == 0:
            return []

        url = f"https://www.ncbi.nlm.nih.gov/research/litvar2-api/variant/search/gene/{gene_symbol}"
        response_text: str = cls._requests_get_text_limited(url)  # type: ignore

        if not response_text or len(response_text) == 0:
            return []

        # This API returns a string of JSON-like objects containing both single-quotes and multiple lines.
        # It's necessary to reformat the string to be valid JSON before parsing.
        response_text = response_text.replace('"', '"')
        response_text = response_text.replace("'", '"')
        return [json.loads(s) for s in response_text.split("\n")]

    @classmethod
    def _validate_variant_id(cls, variant_id: str) -> bool:
        if (
            re.match(cls.LITVAR_RSID_RE, variant_id)
            or re.match(cls.LITVAR_HGVS_RE, variant_id)
            or re.match(cls.LITVAR_CLINGEN_RSID_RE, variant_id)
        ):
            return True
        else:
            return False

    @classmethod
    def pmids_for_variant(cls, variant_id: str) -> Mapping[str, Any]:
        """Return a list of pmids and pmcids in which `variant_id` appears.

        Variant id is of the form expected by litVar, e.g.,
            - litvar%40rs113488022%23%23 for variants with RSIDs
            - litvar%40%23673%23p.F12V for variants with HGVS only
            - litvar%40CA175337%23rs397507479%23%23 for variants with RSIDs and ClinGen IDs
        """
        # Example usage
        # https://www.ncbi.nlm.nih.gov/research/litvar2-api/variant/get/litvar%40rs113488022%23%23/publications

        if not variant_id or len(variant_id) == 0:
            return {}

        variant_id = variant_id.replace("@", "%40").replace("#", "%23")

        if cls._validate_variant_id(variant_id) is False:
            return {}

        url = f"https://www.ncbi.nlm.nih.gov/research/litvar2-api/variant/get/{variant_id}/publications"
        response_text: str = cls._requests_get_text_limited(url)  # type: ignore
        return json.loads(response_text)