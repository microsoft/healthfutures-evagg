import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, reduce
from typing import Any, Dict, List, Optional, Tuple

import openai
import retry
from openai import AzureOpenAI, OpenAI
from openai.types import CreateEmbeddingResponse
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from lib.config import PydanticYamlModel
from lib.evagg.svc.logging import PROMPT

from .interfaces import IPromptClient

ChatMessages = List[ChatCompletionMessageParam]
logger = logging.getLogger(__name__)


class OpenAIConfig(PydanticYamlModel):
    deployment: str
    endpoint: str
    api_key: str
    api_version: str


class OpenAIClient(IPromptClient):
    _config: OpenAIConfig

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = OpenAIConfig(**config)

    @property
    def _client(self) -> OpenAI:
        return self._get_client_instance()

    @lru_cache
    def _get_client_instance(self) -> OpenAI:
        logger.info("Using Azure OpenAI API")
        # Can configure a default model deployment here, but model is still required in the settings, so there's
        # no point in doing so. Instead we'll just put the deployment from the config into the default settings.
        return AzureOpenAI(
            azure_endpoint=self._config.endpoint,
            api_key=self._config.api_key,
            api_version=self._config.api_version,
        )

    @lru_cache
    def _load_prompt_file(self, prompt_file: str) -> str:
        with open(prompt_file, "r") as f:
            return f.read()

    @retry.retry(openai.RateLimitError, tries=5, delay=5, backoff=2)
    def _generate_completion(self, messages: ChatMessages, settings: Dict[str, Any]) -> str:
        start_ts = time.time()
        prompt_tag = settings.pop("prompt_tag", "prompt")
        completion = self._client.chat.completions.create(messages=messages, **settings)
        response = completion.choices[0].message.content or ""
        elapsed = time.time() - start_ts

        prompt_log = {
            "prompt_tag": prompt_tag,
            "prompt_model": f"{settings.get('model')} {self._config.api_version}",
            "prompt_settings": settings,
            "prompt_text": "\n".join([str(m.get("content")) for m in messages]),
            "prompt_response": response,
        }

        logger.log(PROMPT, f"Chat '{prompt_tag}' complete in {elapsed:.2f} seconds.", extra=prompt_log)
        return response

    def prompt(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        params: Optional[Dict[str, str]] = None,
        prompt_settings: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Get the response from a prompt."""
        # Replace any '{{${key}}}' instances with values from the params dictionary.
        user_prompt = reduce(lambda x, kv: x.replace(f"{{{{${kv[0]}}}}}", kv[1]), (params or {}).items(), user_prompt)

        messages: ChatMessages = [ChatCompletionUserMessageParam(role="user", content=user_prompt)]
        if system_prompt:
            messages.insert(0, ChatCompletionSystemMessageParam(role="system", content=system_prompt))

        settings = {
            "max_tokens": 1024,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "temperature": 0.25,
            "model": self._config.deployment,
            **(prompt_settings or {}),
        }

        return self._generate_completion(messages, settings)

    def prompt_file(
        self,
        user_prompt_file: str,
        system_prompt: Optional[str] = None,
        params: Optional[Dict[str, str]] = None,
        prompt_settings: Optional[Dict[str, Any]] = None,
    ) -> str:
        user_prompt = self._load_prompt_file(user_prompt_file)
        return self.prompt(user_prompt, system_prompt, params, prompt_settings)

    def embeddings(
        self, inputs: List[str], embedding_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[float]]:
        settings = {"model": "text-embedding-ada-002", **(embedding_settings or {})}

        start_overall = time.time()
        with ThreadPoolExecutor(max_workers=20) as executor:
            results = executor.map(lambda input: _run_single_embedding(self._client, input, settings), inputs)
        elapsed = time.time() - start_overall

        embeddings = {}
        tokens_used = 0
        for input, embedding, token_count in results:
            embeddings[input] = embedding
            tokens_used += token_count

        logger.info(f"Total of {len(inputs)} embeddings produced in {elapsed:.2f} seconds using {tokens_used} tokens.")
        return embeddings


@retry.retry(openai.RateLimitError, tries=3, delay=5, backoff=2)
def _run_single_embedding(
    openai_client: OpenAI,
    input: str,
    settings: Any,
) -> Tuple[str, List[float], int]:
    result: CreateEmbeddingResponse = openai_client.embeddings.create(input=[input], **settings)
    return (input, result.data[0].embedding, result.usage.prompt_tokens)
