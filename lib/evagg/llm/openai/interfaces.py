from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class OpenAIClientResponse:
    result: Dict[str, Any]
    settings: Dict[str, Any]
    output: str


# TODO, handle logging better?
# TODO, maybe return a response string instead of the OpenAIClientResponse
# TODO, tests


@dataclass
class OpenAIClientEmbeddings:
    embeddings: Dict[str, List[float]]
    response: OpenAIClientResponse

    # dictionary getter
    def __getitem__(self, key: str) -> List[float]:
        return self.embeddings[key]


class IOpenAIClient(Protocol):
    def run(self, prompt: str, settings: Optional[Dict[str, Any]] = None) -> OpenAIClientResponse:
        ...

    def run_file(
        self, prompt_file: str, params: Optional[Dict[str, str]] = None, settings: Optional[Dict[str, Any]] = None
    ) -> OpenAIClientResponse:
        ...

    def chat(self, messages: List[Dict[str, str]], settings: Optional[Dict[str, Any]] = None) -> OpenAIClientResponse:
        ...

    def chat_oneshot(
        self, user_prompt: str, system_prompt: Optional[str] = None, settings: Optional[Dict[str, Any]] = None
    ) -> OpenAIClientResponse:
        ...

    def chat_oneshot_file(
        self,
        user_prompt_file: str,
        system_prompt: Optional[str],
        params: Optional[Dict[str, str]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> OpenAIClientResponse:
        ...

    def embeddings(self, inputs: List[str], settings: Optional[Dict[str, Any]] = None) -> OpenAIClientEmbeddings:
        ...
