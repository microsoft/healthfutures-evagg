import logging
import re
from functools import cache
from typing import Any, Dict, Sequence

from Bio.SeqUtils import IUPACData
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
        # Mutalyzer doesn't currently support normalizing frame shift variants, so we can't back-translate them.
        if hgvsp.split(":")[1].find("fs") != -1:
            logger.debug(f"Skipping back-translation of frameshift variant: {hgvsp}")
            return []

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

        # Only return a subset of the fields in the response.
        response_dict = {}
        if "normalized_description" in response:
            response_dict["normalized_description"] = response["normalized_description"]
        if "protein" in response and "description" in response["protein"]:
            response_dict["protein"] = {"description": response["protein"]["description"]}
        return response_dict

    @cache
    def _normalize_frame_shift(self, hgvs: str) -> Dict[str, Any]:
        """Normalize a frame shift variant using a custom approach."""
        # fs variants are of any of the following forms
        # - p.XNNNfs
        # - p.XNNNYfs
        # - p.XNNNYfs*
        # - p.XNNNYfs*7
        #
        # X and Y can be a one-letter or three-letter amino acid, N is a number, and * is a stop codon,
        # which can alternatively be expressed as Ter.
        #
        # p.(XNNNfs) and p.XNNNfs are both acceptable, the normalized form should retain parentheses.
        #
        # The normalized representation should be p.XNNNfs where X is the three letter amino acid code.

        logger.debug(f"Normalizing frame shift variant {hgvs}")

        refseq, hgvs_desc = hgvs.split(":")

        # Drop anything past NNN and replace with fs.
        hgvs_desc = re.sub(r"(\(?)([A-Za-z]+[0-9]+)[A-Za-z0-9\*]+(\)?)", r"\1\2fs\3", hgvs_desc)

        # Now replace the single letter code with the three letter code, if that's what was used.
        if matched := re.match(r"(p.\(?)([A-Z])([0-9]+fs\)?)", hgvs_desc):
            hgvs_desc = matched.group(1) + IUPACData.protein_letters_1to3[matched.group(2)] + matched.group(3)

        # return {"normalized_description": f"{refseq}:{hgvs_desc}", "protein": {"description": f"{refseq}:{hgvs_desc}"}}
        return {"normalized_description": f"{refseq}:{hgvs_desc}"}

    def normalize(self, hgvs: str) -> Dict[str, Any]:
        """Normalize an HGVS description using Mutalyzer.

        hgvs: The HGVS description to normalize, e.g., NM_000551.3:c.1582G>A
        """
        # Mutalyzer doesn't currently support normalizing frame shift variants, so we have to take our own approach
        # here.
        if hgvs.split(":")[1].find("fs") != -1:
            return self._normalize_frame_shift(hgvs)

        return self._cached_normalize(hgvs)

    def validate(self, hgvs: str) -> bool:
        """Validate an HGVS description using Mutalyzer."""
        # Mutalyzer doesn't currently support normalizing frame shift variants, so we can't validate them.
        # TODO, consider tweaking to be a stop gain and normalizing that.
        if hgvs.split(":")[1].find("fs") != -1:
            logger.debug(f"Skipping validation of frame shift variant {hgvs}")
            return False

        return bool(self.normalize(hgvs))
