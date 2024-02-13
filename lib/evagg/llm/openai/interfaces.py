from typing import Any, Dict, List, Optional, Protocol


class IPromptClient(Protocol):
    def prompt(
        self,
        user_prompt_file: str,
        system_prompt: Optional[str],
        params: Optional[Dict[str, str]] = None,
        prompt_settings: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Get the response from a prompt."""
        ...

    def prompt_file(
        self,
        user_prompt_file: str,
        system_prompt: Optional[str],
        params: Optional[Dict[str, str]] = None,
        prompt_settings: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Get the response from a prompt."""
        ...

    def embeddings(
        self, inputs: List[str], embedding_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[float]]:
        """Get embeddings for the given inputs."""
        ...
