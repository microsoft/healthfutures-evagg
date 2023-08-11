import semantic_kernel as sk

SKILLS_DIRECTORY = "skills"


class FunctionRunner:
    _kernel: sk.Kernel

    def __init__(self, kernel: sk.Kernel, skill_name: str, function_name: str) -> None:
        self._kernel = kernel
        self._func = self._get_function_from_file(kernel, skill_name, function_name)

    def _get_function_from_file(self, kernel: sk.Kernel, skill_name: str, function_name: str) -> sk.SKFunctionBase:
        """Get a function from a skill in a directory."""
        skill = kernel.import_semantic_skill_from_directory(SKILLS_DIRECTORY, skill_name)
        return skill[function_name]

    def run(self, context_variables: sk.ContextVariables) -> sk.SKContext:
        """Run the function."""
        # Naughty, but sk.ContextVariables doesn't provide a way to get the full context without knowing the keys
        print(f"Calling {self._func.name} with the following context\n===\n{context_variables._variables}\n===")
        return self._func.invoke(variables=context_variables)
