import os
from functools import cache

from Bio import Entrez
from dotenv import load_dotenv

from lib.config import PydanticYamlModel

from ._interfaces import IEntrezClient


# Assume AOAI.
class BioEntrezConfig(PydanticYamlModel):
    api_key: str | None
    email: str


class BioEntrezDotEnvConfig(BioEntrezConfig):
    def __init__(self) -> None:
        load_dotenv()
        super().__init__(
            api_key=os.environ["NCBI_EUTILS_API_KEY"] if "NCBI_EUTILS_API_KEY" in os.environ else None,
            email=os.environ["NCBI_EUTILS_EMAIL"],
        )


class BioEntrezClient(IEntrezClient):
    def __init__(self, config: BioEntrezConfig) -> None:
        if config.api_key:
            Entrez.api_key = config.api_key
        Entrez.email = config.email

    @cache
    def efetch(self, db: str, id: str, retmode: str | None = None, rettype: str | None = None) -> str:
        return Entrez.efetch(db=db, id=id, retmode=retmode, rettype=rettype).read()
