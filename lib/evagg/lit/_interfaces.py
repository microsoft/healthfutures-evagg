from typing import Any, Protocol

from lib.evagg import Paper


class IAnnotateEntities(Protocol):
    def annotate(self, paper: Paper) -> dict[str, Any]:
        ...
