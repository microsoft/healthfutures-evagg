import logging
from functools import cache
from typing import Any, Dict, Sequence

from requests.exceptions import HTTPError

from lib.evagg.svc import IWebContentClient

from .interfaces import IBackTranslateVariants, INormalizeVariants

logger = logging.getLogger(__name__)


class MutalyzerClient(INormalizeVariants, IBackTranslateVariants):
    _web_client: IWebContentClient

    def __init__(self, web_client: IWebContentClient) -> None:
        self._web_client = web_client

    @cache
    def _cached_back_translate(self, hgvsp: str) -> Sequence[str]:
        """Back translate a protein variant description to a coding variant description using Mutalyzer.

        hgvsp: The protein variant description to back translate. Must conform to HGVS nomenclature.
        """
        url = f"https://mutalyzer.nl/api/back_translate/{hgvsp}"
        return self._web_client.get(url, "json")

    def back_translate(self, hgvsp: str) -> Sequence[str]:
        return self._cached_back_translate(hgvsp)

    @cache
    def _cached_normalize(self, hgvs: str) -> Dict[str, Any]:
        url = f"https://mutalyzer.nl/api/normalize/{hgvs}"

        # Response code of 422 signifies an unprocessable entity.
        # This occurs when the description is syntactically invalid, but also
        # occurs when the description is biologically invalid (e.g., the reference is incorrect).
        # Detailed information is available in the response body, but it's not currently relevant.
        #
        # Additionally, for at least one variant "NP_000099.2:p.R316X", Mutalyzer returns a 500 error.
        # We do not have additional information why this is being returned, but in either case we need to handle it
        # sensibly, current approach is to return an empty dictionary.
        #
        # TODO: leverage error handling within the web_client itself
        try:
            response = self._web_client.get(url, "json")
        except HTTPError as e:
            if e.response.status_code == 422:
                return {}
            if e.response.status_code == 500:
                return {}
            raise e

        if "errors" in response:
            return {}
        return response

    def normalize(self, hgvs: str) -> Dict[str, Any]:
        """Normalize an HGVS description using Mutalyzer.

        hgvs: The HGVS description to normalize, e.g., NM_000551.3:c.1582G>A
        """
        return self._cached_normalize(hgvs)
