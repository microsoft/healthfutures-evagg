import asyncio
import os
from typing import Dict, Tuple

import semantic_kernel as sk
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import AzureTextCompletion

from lib.config import PydanticYamlModel


# Assume AOAI.
class SemanticKernelConfig(PydanticYamlModel):
    deployment: str
    endpoint: str
    api_key: str
    skill_dir: str | None


class SemanticKernelDotEnvConfig(SemanticKernelConfig):
    def __init__(self) -> None:
        load_dotenv()
        # Assume AOAI.
        super().__init__(
            deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            skill_dir=os.environ.get("SEMANTIC_KERNEL_SKILL_DIR", None),
        )


class SemanticKernelClient:
    DEFAULT_SKILL_DIR = os.path.join(os.path.dirname(__file__), "skills")
    _skill_dir: str = DEFAULT_SKILL_DIR
    _functions: Dict[Tuple[str, str], sk.SKFunctionBase]

    def __init__(self, config: SemanticKernelConfig) -> None:
        self._kernel = sk.Kernel()
        self._functions = {}
        # Assume AOAI.
        self._kernel.add_text_completion_service(
            "completion",
            AzureTextCompletion(
                deployment_name=config.deployment,
                endpoint=config.endpoint,
                api_key=config.api_key,
            ),
        )
        if config.skill_dir:
            self._skill_dir = config.skill_dir

    def _import_function(self, skill: str, function: str) -> sk.SKFunctionBase:
        skill_obj = self._kernel.import_semantic_skill_from_directory(self._skill_dir, skill)
        if function not in skill_obj:
            raise ValueError(f"Function {function} not found in skill {skill}")
        return skill_obj[function]

    def _get_function(self, skill: str, function: str) -> sk.SKFunctionBase:
        key = (skill, function)
        if key in self._functions:
            return self._functions[key]
        else:
            self._functions[key] = self._import_function(skill, function)
            return self._functions[key]

    def _invoke_function(
        self, sk_function: sk.SKFunctionBase, input: str | None, variables: sk.ContextVariables
    ) -> str:
        result = asyncio.run(self._kernel.run_async(sk_function, input_vars=variables, input_str=input))
        if result.error_occurred:
            raise ValueError(f"Error: {result.last_error_description}")
        return result.result

    def run_completion_function(self, skill: str, function: str, context_variables: Dict[str, str]) -> str:
        # TODO handle errors
        # TODO handle token limits?

        function_obj = self._get_function(skill, function)
        input = context_variables.pop("input", None)
        vars = sk.ContextVariables(variables=context_variables)
        return self._invoke_function(function_obj, input, vars)
