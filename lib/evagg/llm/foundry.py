import asyncio
import json
import logging
import time
from functools import lru_cache, reduce
from typing import Any, Dict, Iterable, List, Optional, Tuple

from azure.ai.inference.aio import ChatCompletionsClient
from azure.ai.inference.models import ChatRequestMessage, SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from pydantic import BaseModel

from lib.evagg.utils.cache import ObjectCache
from lib.evagg.utils.logging import PROMPT

from .interfaces import IPromptClient

logger = logging.getLogger(__name__)


class ChatMessages:
    _messages: List[ChatRequestMessage]

    @property
    def content(self) -> str:
        return "".join([json.dumps(message) for message in self._messages])

    def __init__(self, messages: Iterable[ChatRequestMessage]) -> None:
        self._messages = list(messages)

    def __hash__(self) -> int:
        return hash(self.content)

    def insert(self, index: int, message: ChatRequestMessage) -> None:
        self._messages.insert(index, message)

    def to_list(self) -> List[ChatRequestMessage]:
        return self._messages.copy()


class AzureFoundryConfig(BaseModel):
    endpoint: str
    api_key: str
    max_parallel_requests: int = 0


class AzureFoundryClient(IPromptClient):
    _config: AzureFoundryConfig

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = AzureFoundryConfig(**config)

    @property
    def _client(self) -> ChatCompletionsClient:
        return self._get_client_instance()

    @lru_cache
    def _get_client_instance(self) -> ChatCompletionsClient:
        logger.info(
            f"Using Azure Foundry at {self._config.endpoint}" + f" (max_parallel={self._config.max_parallel_requests})."
        )
        return ChatCompletionsClient(
            endpoint=self._config.endpoint,
            credential=AzureKeyCredential(self._config.api_key),
        )

    @lru_cache
    def _load_prompt_file(self, prompt_file: str) -> str:
        with open(prompt_file, "r") as f:
            return f.read()

    def _create_completion_task(self, messages: ChatMessages, settings: Dict[str, Any]) -> asyncio.Task:
        """Schedule a completion task to the event loop and return the awaitable."""
        chat_completion = self._client.complete(messages=messages.to_list(), **settings)
        return asyncio.create_task(chat_completion, name="chat")

    def _parse_raw_response(self, response_raw: str, settings: Dict[str, Any]) -> Tuple[str, str]:
        """Parse the raw response from the chat completion."""
        # First, check whether there's exactly zero or one of each <think> and </think> tags.
        # If there are more than one of either, raise an error.

        # Count the number of <think> and </think> tags in the response.
        think_count = response_raw.count("<think>")
        end_think_count = response_raw.count("</think>")
        if think_count not in [0, 1] or end_think_count not in [0, 1] or think_count != end_think_count:
            raise ValueError(
                f"Invalid response format: {think_count} <think> tags and {end_think_count} </think> tags found."
            )

        if think_count == 0:
            return response_raw, ""

        # If the response contains thinking tokens, they will be prefixed with <think> and suffixed with </think>.
        # Grab everything between <think> and </think>.
        # If no thinking tokens are found, set the result to an empty string.
        thinking_tokens = response_raw.split("<think>")[1].split("</think>")[0]
        thinking_tokens = thinking_tokens.strip()

        # The actual response is everything after the </think> tag.
        response_tokens = response_raw.split("</think>")[1]
        response_tokens = response_tokens.strip()

        # If we're expexting a JSON object, remove the leading ```json and trailing ``` from the response.
        if settings.get("response_format") == "json_object":
            if response_tokens.startswith("```json"):
                response_tokens = response_tokens.split("```json")[1]
            if response_tokens.endswith("```"):
                response_tokens = response_tokens.split("```")[0]
            response_tokens = response_tokens.strip()

        return response_tokens, thinking_tokens

    async def _generate_completion(self, messages: ChatMessages, settings: Dict[str, Any]) -> str:
        prompt_tag = settings.pop("prompt_tag", "prompt")
        prompt_metadata = settings.pop("prompt_metadata", {})

        while True:
            try:
                # Pause 1 second if the number of pending chat completions is at the limit.
                if (max_requests := self._config.max_parallel_requests) > 0:
                    while sum(1 for t in asyncio.all_tasks() if t.get_name() == "chat") > max_requests:
                        await asyncio.sleep(1)

                start_ts = time.time()
                completion = await self._create_completion_task(messages, settings)
                response_raw = completion.choices[0].message.content or ""
                response, thought = self._parse_raw_response(response_raw, settings)
                elapsed = time.time() - start_ts
                break
            except Exception as e:
                import pdb

                pdb.set_trace()
                logger.warning(f"Error on {prompt_tag}: {e}")
                await asyncio.sleep(1)

        prompt_metadata["returned_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        prompt_metadata["elapsed_time"] = f"{elapsed:.1f} seconds"

        prompt_log = {
            "prompt_tag": prompt_tag,
            "prompt_metadata": prompt_metadata,
            "prompt_settings": settings,
            "prompt_text": "\n".join([str(m.get("content")) for m in messages.to_list()]),
            "prompt_thought": thought,
            "prompt_response": response,
        }

        logger.log(PROMPT, f"Chat '{prompt_tag}' complete in {elapsed:.1f} seconds.", extra=prompt_log)
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

        messages: ChatMessages = ChatMessages([UserMessage(content=user_prompt)])
        if system_prompt:
            messages.insert(0, SystemMessage(content=system_prompt))

        settings = {
            "max_tokens": 1024,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "temperature": 0.7,
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

    async def embeddings(
        self, inputs: List[str], embedding_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[float]]:
        raise NotImplementedError("Embeddings are not supported in Azure Foundry.")


class AzureFoundryCacheClient(AzureFoundryClient):
    def __init__(self, config: Dict[str, Any]) -> None:
        # Don't cache tasks that have errored out.
        self.task_cache = ObjectCache[asyncio.Task](lambda t: not t.done() or t.exception() is None)
        super().__init__(config)

    def _create_completion_task(self, messages: ChatMessages, settings: Dict[str, Any]) -> asyncio.Task:
        """Create a new task only if no identical non-errored one was already cached."""
        cache_key = hash((messages.content, json.dumps(settings)))
        if task := self.task_cache.get(cache_key):
            return task
        task = super()._create_completion_task(messages, settings)
        self.task_cache.set(cache_key, task)
        return task
