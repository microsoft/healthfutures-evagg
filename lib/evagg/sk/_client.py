import os
from typing import Tuple

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureTextCompletion

from lib.config import PydanticYamlModel


class SemanticKernelConfig(PydanticYamlModel):
    deployment: str
    endpoint: str
    api_key: str


class SemanticKernelClient:
    _SKILL_DIR = "lib/evagg/skills"
    _functions: dict[Tuple[str, str], sk.SKFunctionBase] = {}

    def __init__(self, config: SemanticKernelConfig) -> None:
        self._kernel = sk.Kernel()
        self._kernel.add_text_completion_service(
            "completion",
            AzureTextCompletion(
                deployment_name=config.deployment,
                endpoint=config.endpoint,
                api_key=config.api_key,
            ),
        )

    def _resolve_skill_dir(self) -> str:
        return os.path.join(os.path.dirname(__file__), self._SKILL_DIR)

    def _get_function(self, skill: str, function: str) -> sk.SKFunctionBase:
        key = (skill, function)
        if key in self._functions:
            return self._functions[key]
        else:
            skill_obj = self._kernel.import_semantic_skill_from_directory(self._resolve_skill_dir(), skill)
            if function not in skill_obj:
                raise ValueError(f"Function {function} not found in skill {skill}")
            self._functions[key] = skill_obj[function]
            return skill_obj[function]

    def run_completion_function(self, skill: str, function: str, context_variables: dict[str, str]) -> str:
        # TODO handle errors
        # TODO handle token limits?
        function_obj = self._get_function(skill, function)
        input = context_variables.pop("input", None)
        vars = sk.ContextVariables(variables=context_variables)
        result = function_obj.invoke(input=input, variables=vars)
        if result.error_occurred:
            raise ValueError(f"Error running function {function}: {result.last_error_description}")
        return result.result
