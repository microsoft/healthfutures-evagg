import logging
from functools import cache
from typing import Any, Dict, Sequence

from requests.exceptions import HTTPError

from lib.evagg.svc import IWebContentClient

from .interfaces import IBackTranslateVariants, INormalizeVariants, IValidateVariants

logger = logging.getLogger(__name__)


class MutalyzerClient(INormalizeVariants, IBackTranslateVariants, IValidateVariants):
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
        # Preprocess the hgvs string to avoid some common issues.
        # Remove whitespace and unicode whitespace
        hgvs = hgvs.replace(" ", "").replace("\u2009", "")

        # Response code of 422 signifies an unprocessable entity. This occurs when the description is syntactically
        # invalid, but also occurs when the description is biologically invalid (e.g., the reference is incorrect).
        # Mutalyzer's web client should be configured with no_raise_codes=[422] to avoid raising an exception.
        # For at least one variant (NP_000099.2:p.R316X), Mutalyzer returns a 500 error, which it shouldn't (500
        # is an internal server error). For now we interpret this as an unresolvable entity and return an empty dict.
        url = f"https://mutalyzer.nl/api/normalize/{hgvs}"
        try:
            response = self._web_client.get(url, "json")
        except HTTPError as e:
            logger.debug(f"{url} returned an error: {e}")
            if e.response.status_code == 500:
                return {}
            raise e

        if "errors" in response or ("custom" in response and "errors" in response["custom"]):
            logger.warning(f"Mutalyzer returned an unhandleable error for {hgvs}: {response['custom']['errors']}")
            return {}
        return response

    def normalize(self, hgvs: str) -> Dict[str, Any]:
        """Normalize an HGVS description using Mutalyzer.

        hgvs: The HGVS description to normalize, e.g., NM_000551.3:c.1582G>A
        """
        return self._cached_normalize(hgvs)

    def validate(self, hgvs: str) -> bool:
        """Validate an HGVS description using Mutalyzer."""
        # Mutalyzer doesn't currently support normalizing frame shift variants, so we can't validate them.
        # TODO, consider tweaking to be a stop gain and normalizing that.
        if hgvs.split(":")[1].find("fs") != -1:
            return True

        return bool(self.normalize(hgvs))
