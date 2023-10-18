from typing import Dict, Protocol


class ISemanticKernelClient(Protocol):
    def run_completion_function(self, skill: str, function: str, context_variables: Dict[str, str]) -> str:
        ...
