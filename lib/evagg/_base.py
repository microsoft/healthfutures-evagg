# TODO dataclass?
# TODO should be immutable after load.
class Variant:
    def __init__(self, gene: str, modification: str) -> None:
        self._gene = gene
        self._modification = modification


# TODO dataclass?
# TODO should be immutable after load.
class Paper:
    def __init__(self, id: str, citation: str, abstract: str) -> None:
        self.id = id
        self.citation = citation
        self.abstract = abstract

    def __repr__(self) -> str:
        m = 10
        asuf = "..." if len(self.abstract) > m else ""

        return f'id: {self.id} - abstract: "{self.abstract[:m]}{asuf}"'

    @classmethod
    def from_dict(cls, values: dict[str, str]) -> "Paper":
        return Paper(id=values["id"], citation=values["citation"], abstract=values["abstract"])
