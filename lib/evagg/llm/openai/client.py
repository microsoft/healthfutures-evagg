import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import openai
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from retry import retry as Retry  # noqa

from lib.config import PydanticYamlModel

from .interfaces import IOpenAIClient, OpenAIClientEmbeddings, OpenAIClientResponse

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

    # TODO, return type?
    def _completion_create(self, settings: Dict[str, Any]) -> Any:
        start_ts = time.time()
        settings = self._clean_completion_settings(settings)
        result = self._client.completions.create(**settings)  # type: ignore
        logger.info(f"Completion took {time.time() - start_ts} seconds")
        return result

    # TODO, return type?
    def _chat_completion_create(self, settings: Dict[str, Any]) -> Any:
        start_ts = time.time()
        settings = self._clean_completion_settings(settings)
        result = self._client.chat.completions.create(**settings)  # type: ignore
        logger.info(f"Chat completion took {time.time() - start_ts} seconds")
        return result

    def run(self, prompt: str, settings: Optional[Dict[str, Any]] = None) -> OpenAIClientResponse:
        default_settings = {
            "max_tokens": 1024,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": ["</result>", "[END]", "[DONE]", "#END#"],
            "temperature": 0.25,
            "model": self._config.deployment,
        }

        # merge settings
        if settings:
            settings = {**default_settings, **settings}
        else:
            settings = default_settings

        settings["prompt"] = prompt

        logger.info(f"Running completion with settings: {settings}")

        result = self._completion_create(settings)
        output = result.choices[0].text.strip()

        logger.info(f"Completion output: {output}")

        return OpenAIClientResponse(result=result, settings=settings, output=output)

    def run_file(
        self, prompt_file: str, params: Optional[Dict[str, str]] = None, settings: Optional[Dict[str, Any]] = None
    ) -> OpenAIClientResponse:
        prompt = self._load_prompt_file(prompt_file)
        if params:
            for key, value in params.items():
                prompt = prompt.replace(f"{{{{${key}}}}}", value)

        return self.run(prompt, settings)

    def chat(
        self,
        messages: List[Dict[str, str]],
        settings: Optional[Dict[str, Any]] = None,
    ) -> OpenAIClientResponse:
        default_settings = {
            "max_tokens": 1024,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "temperature": 0.25,
            "messages": [],
            "model": self._config.deployment,
        }

        # merge settings
        if settings:
            settings = {**default_settings, **settings}
        else:
            settings = default_settings

        settings["messages"] = messages

        logger.debug(f"Running chat completion with settings: {settings}")

        result = self._chat_completion_create(settings)
        output = result.choices[0].message.content

        logger.debug(f"Chat completion output: {output}")

        return OpenAIClientResponse(result=result, settings=settings, output=output)

    def chat_oneshot(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> OpenAIClientResponse:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        return self.chat(messages, settings)

    def chat_oneshot_file(
        self,
        user_prompt_file: str,
        system_prompt: Optional[str],
        params: Optional[Dict[str, str]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> OpenAIClientResponse:
        user_prompt = self._load_prompt_file(user_prompt_file)

        if params:
            for key, value in params.items():
                user_prompt = user_prompt.replace(f"{{{{${key}}}}}", value)

        return self.chat_oneshot(user_prompt=user_prompt, system_prompt=system_prompt, settings=settings)

    def embeddings(self, inputs: List[str], settings: Optional[Dict[str, Any]] = None) -> OpenAIClientEmbeddings:
        default_settings = {
            "model": "text-embedding-ada-002",
        }

        # merge settings
        if settings:
            settings = {**default_settings, **settings}
        else:
            settings = default_settings

        return self._run_parallel_embeddings(
            inputs=inputs,
            settings=settings,
        )

    def _run_parallel_embeddings(self, inputs: List[str], settings: Any) -> OpenAIClientEmbeddings:
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
        settings["prompt"] = "\n".join(inputs)
        response = OpenAIClientResponse(result=merged_result, settings=settings, output="")

        return OpenAIClientEmbeddings(embeddings=embeddings, response=response)

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
@Retry(openai.RateLimitError, tries=3, delay=5, backoff=2)
def _run_single_embedding(
    openai_client: OpenAI,
    input: str,
    settings: Any,
) -> Tuple[str, Any]:
    start_ts = time.time()

    result = openai_client.embeddings.create(input=[input], **settings)  # type: ignore

    logger.info(f"Embedding took {time.time() - start_ts} seconds")

    return (input, result)
