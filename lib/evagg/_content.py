from typing import Sequence

from ._base import Paper
from ._interfaces import IExtractFields


class SimpleContentExtractor(IExtractFields):
    def __init__(self, fields: Sequence[str]) -> None:
        self._fields = fields

    def _field_to_value(self, field: str) -> str:
        if field == "gene":
            return "CHI3L1"
        if field == "variant":
            return "p.Y34C"
        if field == "MOI":
            return "AD"
        if field == "phenotype":
            return "Long face (HP:0000276)"
        if field == "functional data":
            return "No"
        else:
            return "Unknown"

    def extract(self, query: Query, paper: Paper) -> Sequence[dict[str, str]]:
        # Dummy implementation that returns a single variant with a static set of fields.
        return [{field: self._field_to_value(field) for field in self._fields}]


from typing import Protocol

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureTextCompletion

from lib.config import PydanticYamlModel


class SemanticKernelConfig(PydanticYamlModel):
    deployment: str
    endpoint: str
    api_key: str


class ISemanticKernelClient(Protocol):
    def run_completion_function(self, skill: str, function: str, context: dict[str, str]) -> str:
        ...


class IExtractEntities(Protocol):
    def extract_variants(self, query: Query, paper: Paper) -> dict[str, Sequence[str]]:
        """Extract variants relevant to query that are mentioned in `paper`.

        Returns a dictionary mapping each variant to a list of text chunks that mention it.
        """
        ...


class SemanticKernelContentExtractor(IExtractFields):
    def __init__(
        self, fields: Sequence[str], sk_client: ISemanticKernelClient, entity_extractor: IExtractEntities
    ) -> None:
        self._fields = fields
        self._sk_client = sk_client
        self._entity_extractor = entity_extractor

        # self._kernel = sk.Kernel()
        # self._kernel.add_text_completion_service(
        #     "completion",
        #     AzureTextCompletion(
        #         deployment_name=config.deployment,
        #         endpoint=config.endpoint,
        #         api_key=config.api_key,
        #     ),
        # )

    def extract(self, query: Query, paper: Paper) -> Sequence[dict[str, str]]:
        # Find all the variant mentions in the paper
        variants = self._entity_extractor.extract_variants(query, paper)

        # For each variant/field pair, run the appropriate prompt.
        overall_result = []
        for variant in variants:
            variant_result = {}
            for field in self._fields:
                result = self._sk_client.run_completion_function(
                    skill="content", function=field, context={"variant": variant, "gene": query._gene}
                )
                variant_result[field] = result
            overall_result.append(variant_result)
        return overall_result
