import semantic_kernel as sk
import json


def context_variables_from_file(filepath: str) -> sk.ContextVariables:
    """Read parameters from a file."""
    with open(filepath, 'r') as f:
        parameters = json.load(f)
        variables = {k: v for k, v in parameters.items() if k != "input"}
        return sk.ContextVariables(content=parameters["input"], variables=variables)
