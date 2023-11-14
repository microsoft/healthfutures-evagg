import os
from functools import cache
from typing import Dict

from Bio import Entrez
from dotenv import load_dotenv

from lib.config import PydanticYamlModel

from ._interfaces import IEntrezClient


# Assume AOAI.
class BioEntrezConfig(PydanticYamlModel):
    api_key: str | None
    email: str
    max_tries: str = "10"


class BioEntrezDotEnvConfig(BioEntrezConfig):
    def __init__(self) -> None:
        load_dotenv()

        bag: Dict[str, str] = {}
        for k, v in os.environ.items():
            if k.startswith("NCBI_EUTILS_"):
                new_key = k.replace("NCBI_EUTILS_", "").lower()
                bag[new_key] = v
        super().__init__(**bag)


class BioEntrezClient(IEntrezClient):
    def __init__(self, config: BioEntrezConfig) -> None:
        if config.api_key:
            Entrez.api_key = config.api_key

        # This isn't particularly clear from the documentation, but it looks like
        # we're getting 400s when max_tries is set too low.
        # see https://biopython.org/docs/1.75/api/Bio.Entrez.html
        Entrez.max_tries = int(config.max_tries)
        Entrez.email = config.email

    @cache
    def efetch(self, db: str, id: str, retmode: str | None = None, rettype: str | None = None) -> str:
        return Entrez.efetch(db=db, id=id, retmode=retmode, rettype=rettype).read()

    @cache
    def esearch(self, db: str, term: str, sort: str, retmax: int, retmode: str | None = None) -> str:
        return Entrez.esearch(db=db, term=term, sort=sort, retmax=retmax, retmode=retmode).read()