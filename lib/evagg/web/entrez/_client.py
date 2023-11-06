from functools import cache

from Bio import Entrez

from ._interfaces import IEntrezClient


class BioEntrezClient(IEntrezClient):
    def __init__(self, email: str):
        Entrez.email = email

    @cache
    def efetch(self, db: str, id: str, retmode: str | None = None, rettype: str | None = None) -> str:
        return Entrez.efetch(db=db, id=id, retmode=retmode, rettype=rettype).read()
