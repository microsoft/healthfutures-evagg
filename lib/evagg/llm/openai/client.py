import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import openai
import retry
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from lib.config import PydanticYamlModel
from lib.evagg.svc.logging import PROMPT

from .interfaces import IOpenAIClient, OpenAIClientEmbeddings

ChatCompletionMessages = List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]]

logger = logging.getLogger(__name__)


class OpenAIConfig(PydanticYamlModel):
    deployment: str
    endpoint: str
    api_key: str
    api_version: str


class OpenAIDotEnvConfig(OpenAIConfig):
    _REQUIRED_ENV_VARS = [
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
    ]

    def __init__(self) -> None:
        if not load_dotenv():
            logger.warning("No .env file found, using pre-existing environment variables.")

        if any(var not in os.environ for var in self._REQUIRED_ENV_VARS):
            raise ValueError(
                f"Missing one or more required environment variables: {', '.join(self._REQUIRED_ENV_VARS)}"
            )

        bag: Dict[str, str] = {}
        for k in self._REQUIRED_ENV_VARS:
            new_key = k.replace("AZURE_OPENAI_", "").lower()
            bag[new_key] = os.environ[k]
        super().__init__(**bag)


class OpenAIClientFactory:
    @staticmethod
    def create(config: OpenAIConfig) -> IOpenAIClient:
        return OpenAIClient(config)

    @staticmethod
    def create_from_dict(config: Dict[str, str]) -> IOpenAIClient:
        return OpenAIClientFactory.create(OpenAIConfig.parse_obj(config))

    @staticmethod
    def create_from_json_string(config: str) -> IOpenAIClient:
        return OpenAIClientFactory.create(OpenAIConfig.parse_raw(config))


class OpenAIClient(IOpenAIClient):
    _config: OpenAIConfig

    def __init__(self, config: OpenAIConfig) -> None:
        self._config = config

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

    def _generate_completion(self, messages: ChatCompletionMessages, settings: Dict[str, Any]) -> Optional[str]:
        start_ts = time.time()
        settings = self._clean_completion_settings(settings)
        completion = self._client.chat.completions.create(messages=messages, **settings)  # type: ignore
        response = completion.choices[0].message.content
        prompt_log = {
            "prompt_settings": settings,
            "prompt_text": "\n".join([str(m["content"]) for m in messages]),
            "prompt_response": response,
            # "prompt_key": 
        }
        logger.log(PROMPT, f"Chat complete in {(time.time() - start_ts):.2f} seconds.", extra=prompt_log)
        return response

    def prompt(
        self,
        user_prompt_file: str,
        system_prompt: Optional[str],
        params: Optional[Dict[str, str]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> str:
        user_prompt = self._load_prompt_file(user_prompt_file)

        for key, value in params.items() if params else {}:
            user_prompt = user_prompt.replace(f"{{{{${key}}}}}", value)

        messages: ChatCompletionMessages = [ChatCompletionUserMessageParam(role="user", content=user_prompt)]
        if system_prompt:
            messages.insert(0, ChatCompletionSystemMessageParam(role="system", content=system_prompt))

        settings = {
            "max_tokens": 1024,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "temperature": 0.25,
            "model": self._config.deployment,
            **(settings or {}),
        }

        response = self._generate_completion(messages, settings)
        return response or ""

    def embeddings(self, inputs: List[str], settings: Optional[Dict[str, Any]] = None) -> OpenAIClientEmbeddings:
        settings = {
            "model": "text-embedding-ada-002",
            **(settings or {}),
        }
        start_overall = time.time()
        results = []

        # clean up before executing
        settings = self._clean_embedding_settings(settings)

        # use threadpool to call _run_one_embedding
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(_run_single_embedding, self._client, input, settings) for input in inputs]
            results = [f.result() for f in futures]

        logger.info(f"Overall time for {len(inputs)} embeddings: {time.time() - start_overall}")

        embeddings = {}
        tokens_used = 0
        model = ""

        for input, result in results:
            vector = result.data[0].embedding
            tokens_used += result.usage.prompt_tokens
            model = result.model
            embeddings[input] = vector

        # HACK: this is because embeddings have a different data model than the other endpoints
        merged_result = {
            "usage": {"prompt_tokens": tokens_used, "completion_tokens": 0},
            "model": model,
        }
        return OpenAIClientEmbeddings(embeddings=embeddings, response=merged_result)

    def _clean_completion_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        # make a copy
        settings = dict(settings)

        # if stop tokens are specified, remove any blank ones
        if "stop" in settings:
            stop_tokens = settings["stop"]

            # wrap in list if not already
            if isinstance(stop_tokens, str):
                stop_tokens = [stop_tokens]

            if not isinstance(stop_tokens, list):
                raise ValueError(f"Stop tokens must be a list but is: {type(stop_tokens)}")

            settings["stop"] = [s for s in stop_tokens if s]

            # if there are no stop tokens, remove the key entirely
            if not settings["stop"]:
                del settings["stop"]

        return settings

    def _clean_embedding_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        # make a copy
        settings = dict(settings)

        return settings


# TODO, return type,
@retry.retry(openai.RateLimitError, tries=3, delay=5, backoff=2)
def _run_single_embedding(
    openai_client: OpenAI,
    input: str,
    settings: Any,
) -> Tuple[str, Any]:
    start_ts = time.time()

    result = openai_client.embeddings.create(input=[input], **settings)  # type: ignore

    logger.info(f"Embedding took {time.time() - start_ts} seconds")

    return (input, result)
