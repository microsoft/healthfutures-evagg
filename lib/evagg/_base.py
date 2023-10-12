# TODO dataclass?
# TODO should be immutable after load.
class Query:
    def __init__(self, gene: str, variant: str) -> None:
        self._gene = gene
        self._variant = variant


# TODO dataclass?
# TODO should be immutable after load.
class Paper:
    def __init__(self, id: str, citation: str, abstract: str, pmcid: str) -> None:
        self.id = id
        self.citation = citation
        self.abstract = abstract
        self.pmcid = pmcid

    def __repr__(self) -> str:
        m = 10
        asuf = "..." if len(self.abstract) > m else ""

        return f'id: {self.id} - abstract: "{self.abstract[:m]}{asuf}"'

    @classmethod
    def from_dict(cls, values: dict[str, str]) -> "Paper":
        return Paper(**values)
