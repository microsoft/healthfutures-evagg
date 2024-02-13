from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class OpenAIClientEmbeddings:
    embeddings: Dict[str, List[float]]
    response: Dict[str, Any]

    # dictionary getter
    def __getitem__(self, key: str) -> List[float]:
        return self.embeddings[key]


class IOpenAIClient(Protocol):
    def prompt(
        self,
        user_prompt_file: str,
        system_prompt: Optional[str],
        params: Optional[Dict[str, str]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Get the response from a prompt."""
        ...

    def prompt_file(
        self,
        user_prompt_file: str,
        system_prompt: Optional[str],
        params: Optional[Dict[str, str]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Get the response from a prompt."""
        ...

    def embeddings(self, inputs: List[str], settings: Optional[Dict[str, Any]] = None) -> OpenAIClientEmbeddings:
        """Get embeddings for the given inputs."""
        ...
