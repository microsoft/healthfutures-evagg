import asyncio
import logging
import time
from functools import lru_cache, reduce
from typing import Any, Coroutine, Dict, List, Optional, Tuple

import openai
from openai import AsyncAzureOpenAI, AsyncOpenAI
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
    max_parallel_requests: int = 0


class OpenAIClient(IPromptClient):
    _config: OpenAIConfig

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = OpenAIConfig(**config)

    @property
    def _client(self) -> AsyncOpenAI:
        return self._get_client_instance()

    @lru_cache
    def _get_client_instance(self) -> AsyncOpenAI:
        logger.info(f"Using AOAI API at {self._config.endpoint} (max_parallel={self._config.max_parallel_requests}).")
        # Can configure a default model deployment here, but model is still required in the settings, so there's
        # no point in doing so. Instead we'll just put the deployment from the config into the default settings.
        return AsyncAzureOpenAI(
            azure_endpoint=self._config.endpoint,
            api_key=self._config.api_key,
            api_version=self._config.api_version,
        )

    @lru_cache
    def _load_prompt_file(self, prompt_file: str) -> str:
        with open(prompt_file, "r") as f:
            return f.read()

    async def _run_timed_completion(self, chat_completion: Coroutine[Any, Any, Any]) -> Tuple[str, float]:
        # Pause 1 second if the number of pending chat completions is at the limit.
        if self._config.max_parallel_requests > 0:
            while sum(1 for t in asyncio.all_tasks() if t.get_name() == "chat") > self._config.max_parallel_requests:
                await asyncio.sleep(1)

        start_ts = time.time()
        completion = await asyncio.create_task(chat_completion, name="chat")
        elapsed = time.time() - start_ts

        return completion.choices[0].message.content or "", elapsed

    async def _generate_completion(self, messages: ChatMessages, settings: Dict[str, Any]) -> str:
        prompt_tag = settings.pop("prompt_tag", "prompt")
        connection_errors = 0

        while True:
            try:
                chat_completion = self._client.chat.completions.create(messages=messages, **settings)
                response, elapsed = await self._run_timed_completion(chat_completion)
                break
            except (openai.RateLimitError, openai.InternalServerError) as e:
                logger.warning(f"Rate limit error on {prompt_tag}: {e}")
                await asyncio.sleep(1)
            except (openai.APIConnectionError, openai.APITimeoutError):
                if connection_errors > 2:
                    if self._config.endpoint.startswith("http://localhost"):
                        logger.error("Azure OpenAI API unreachable - have you started the proxy?")
                    raise
                logger.warning(f"Connectivity error on {prompt_tag}, retrying...")
                connection_errors += 1
                await asyncio.sleep(1)

        prompt_log = {
            "prompt_tag": prompt_tag,
            "prompt_model": f"{settings.get('model')} {self._config.api_version}",
            "prompt_settings": settings,
            "prompt_text": "\n".join([str(m.get("content")) for m in messages]),
            "prompt_response": response,
        }

        logger.log(PROMPT, f"Chat '{prompt_tag}' complete in {elapsed:.2f} seconds.", extra=prompt_log)
        return response

    async def prompt(
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
            "temperature": 0.7,
            "model": self._config.deployment,
            **(prompt_settings or {}),
        }

        return await self._generate_completion(messages, settings)

    async def prompt_file(
        self,
        user_prompt_file: str,
        system_prompt: Optional[str] = None,
        params: Optional[Dict[str, str]] = None,
        prompt_settings: Optional[Dict[str, Any]] = None,
    ) -> str:
        user_prompt = self._load_prompt_file(user_prompt_file)
        return await self.prompt(user_prompt, system_prompt, params, prompt_settings)

    # async def embeddings(
    #     self, inputs: List[str], embedding_settings: Optional[Dict[str, Any]] = None
    # ) -> Dict[str, List[float]]:
    #     settings = {"model": "text-embedding-ada-002", **(embedding_settings or {})}

    #     embeddings = {}

    #     async def _run_single_embedding(input: str) -> int:
    #         result: CreateEmbeddingResponse = await self._client.embeddings.create(input=[input], **settings)
    #         embeddings[input] = result.data[0].embedding
    #         return result.usage.prompt_tokens

    #     start_overall = time.time()
    #     tokens = await asyncio.gather(*[_run_single_embedding(input) for input in inputs])
    #     elapsed = time.time() - start_overall

    #     logger.info(f"{len(inputs)} embeddings produced in {elapsed:.2f} seconds using {sum(tokens)} tokens.")
    #     return embeddings

    async def embeddings(
        self, inputs: List[str], embedding_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[float]]:
        settings = {"model": "text-embedding-ada-002", **(embedding_settings or {})}

        embeddings = {}

        async def _run_single_embedding(input: str) -> int:
            connection_errors = 0
            while True:
                try:
                    result: CreateEmbeddingResponse = await self._client.embeddings.create(input=[input], **settings)
                    embeddings[input] = result.data[0].embedding
                    return result.usage.prompt_tokens
                except (openai.RateLimitError, openai.InternalServerError) as e:
                    logger.warning(f"Rate limit error on embeddings: {e}")
                    await asyncio.sleep(1)
                except (openai.APIConnectionError, openai.APITimeoutError):
                    if connection_errors > 2:
                        if self._config.endpoint.startswith("http://localhost"):
                            logger.error("Azure OpenAI API unreachable - have you started the proxy?")
                        raise
                    logger.warning(f"Connectivity error on embeddings, retrying...")
                    connection_errors += 1
                    await asyncio.sleep(1)

        start_overall = time.time()
        tokens = await asyncio.gather(*[_run_single_embedding(input) for input in inputs])
        elapsed = time.time() - start_overall

        logger.info(f"{len(inputs)} embeddings produced in {elapsed:.2f} seconds using {sum(tokens)} tokens.")
        return embeddings
