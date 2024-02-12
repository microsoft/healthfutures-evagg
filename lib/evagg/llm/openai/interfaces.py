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
        """Run the model with the given prompt."""
        ...

    def run_file(
        self, prompt_file: str, params: Optional[Dict[str, str]] = None, settings: Optional[Dict[str, Any]] = None
    ) -> OpenAIClientResponse:
        """Run the model with the given prompt file."""
        ...

    def chat(self, messages: List[Dict[str, str]], settings: Optional[Dict[str, Any]] = None) -> OpenAIClientResponse:
        """Chat with the model."""
        ...

    def chat_oneshot(
        self, user_prompt: str, system_prompt: Optional[str] = None, settings: Optional[Dict[str, Any]] = None
    ) -> OpenAIClientResponse:
        """Chat with the model, using a single prompt."""
        ...

    def chat_oneshot_file(
        self,
        user_prompt_file: str,
        system_prompt: Optional[str],
        params: Optional[Dict[str, str]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> OpenAIClientResponse:
        """Chat with the model, using a single prompt file."""
        ...

    def embeddings(self, inputs: List[str], settings: Optional[Dict[str, Any]] = None) -> OpenAIClientEmbeddings:
        """Get embeddings for the given inputs."""
        ...

    def get_prompt_response(
        self,
        user_prompt_file: str,
        system_prompt: Optional[str],
        params: Optional[Dict[str, str]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Get the response from a prompt."""
        ...
