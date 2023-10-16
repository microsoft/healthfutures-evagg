from typing import Protocol


class ISemanticKernelClient(Protocol):
    def run_completion_function(self, skill: str, function: str, context_variables: dict[str, str]) -> str:
        ...
