from typing import Any, Dict, List, Optional, Protocol


class IPromptClient(Protocol):
    def prompt(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        params: Optional[Dict[str, str]] = None,
        prompt_settings: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Get the response from a prompt."""
        ...  # pragma: no cover

    def prompt_file(
        self,
        user_prompt_file: str,
        system_prompt: Optional[str] = None,
        params: Optional[Dict[str, str]] = None,
        prompt_settings: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Get the response from a prompt with an input file."""
        ...  # pragma: no cover

    def embeddings(
        self, inputs: List[str], embedding_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[float]]:
        """Get embeddings for the given inputs."""
        ...  # pragma: no cover
