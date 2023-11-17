import json
from typing import Any, Dict, List, Sequence

import requests
from defusedxml import ElementTree

from lib.evagg.sk import ISemanticKernelClient
from lib.evagg.types import IPaperQuery, Paper

from .._interfaces import IExtractFields


class SubjectBasedContentExtractor(IExtractFields):
    _SUPPORTED_FIELDS = {"gene", "paper_id", "hgvsc", "hgvsp", "phenotype", "zygosity", "inheritance"}

    def __init__(self, fields: Sequence[str], sk_client: ISemanticKernelClient) -> None:
        self._fields = fields
        self._sk_client = sk_client

    def _get_paper_content(self, paper: Paper, section_types: Sequence[str]) -> str:
        doc = self._get_paper_xml(paper)

        text = ""

        for p in doc.iter("passage"):
            for i in p.iter("infon"):
                if i.attrib["key"] == "section_type":
                    if i.text in section_types:
                        text += p.find("text").text + "\n\n"
                        break

        return text.__repr__()
        # # The BioC API will by default return unicode text, and sometimes this gives GPT models trouble.
        # return text.encode("unicode-escape").decode()

    def _get_paper_xml(self, paper: Paper, encoding: str = "unicode") -> Any:
        if encoding not in ["unicode", "ascii"]:
            raise ValueError(f"Unexpected encoding: {encoding}")

        pmcid = paper.props["pmcid"]
        if not pmcid.startswith("PMC"):
            print(f"WARNING: Unexpected PMCID format: {pmcid}. prepending with PMC to disambiguate.")
            pmcid = f"PMC{pmcid}"

        url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{pmcid}/{encoding}"
        response = requests.get(url)

        # The above URL supports the following query types:
        # - PMC12345678
        # - 12345678
        #
        # In the case of the former, it will attempt to match a PMCID. In the case of the latter, it will attempt to
        # match the PMID. In both cases, the returned result will be identified by the PMCID.

        # Totally bogus pmcid will result in HTTP 500, not found will result in 404, pmcid with a comma will result
        # in a 400.
        response.raise_for_status()

        root = ElementTree.fromstring(response.content)

        for doc in root.iter("document"):
            id_elt = doc.find("id")
            if id_elt is not None and id_elt.text == pmcid.lstrip("PMC"):
                return doc

        raise ValueError(f"Unexpected result from BioC API, found papers, but none corresponding to {pmcid}")

    def extract(self, paper: Paper, query: IPaperQuery) -> Sequence[Dict[str, str]]:
        # Only process papers in PMC-OA.
        if "pmcid" not in paper.props or paper.props["pmcid"] == "" or paper.props["is_pmc_oa"] != "True":
            print(f"WARNING: Paper {paper.id} is not in PMC-OA, skipping.")
            return []

        # Get the list of subjects discussed.
        section_types = ["TITLE", "RESULTS", "ABSTRACT", "DISCUSS", "METHODS", "TABLE"]
        paper_content = self._get_paper_content(paper, section_types)
        print(f"{paper_content[:100]}...")
        context_variables = {"input": paper_content}

        result_unstructured = self._sk_client.run_completion_function(
            skill="content", function="get_subject_identifiers", context_variables=context_variables
        )
        print(result_unstructured)
        unstructured_context_variables = {"input": result_unstructured}
        subject_identifiers_unparsed = self._sk_client.run_completion_function(
            skill="content", function="structure_subject_identifiers", context_variables=unstructured_context_variables
        )
        print(subject_identifiers_unparsed)

        # TODO, handle json parsing errors.
        subject_identifiers = json.loads(subject_identifiers_unparsed)

        print(subject_identifiers)

        import pdb

        pdb.set_trace()

        return []
