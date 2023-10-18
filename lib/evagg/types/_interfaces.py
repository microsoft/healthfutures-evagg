from typing import Protocol, Set

from ._base import Variant


class IPaperQuery(Protocol):
    def terms(self) -> Set[Variant]:
        ...
