

from model.kernel import get_kernel_from_env
from model.function import FunctionRunner
from model.context import context_variables_from_file

import click


@click.command()
@click.option("--parameters", default="./test/data/lrs-moi-tubb3.json", type=click.Path(exists=True), help="Path to json-formatted parameters file, these should match the parameters in the target function")
@click.option("--skill", default="LitReaderSkill", help="Skill to use (default LitReaderSkill)")
@click.option("--func", default="ModeOfInheritance", help="Function to run (default ModeOfInheritance)")
def main(parameters: str, skill: str, func: str) -> None:
    runner = FunctionRunner(get_kernel_from_env(), skill, func)
    context = context_variables_from_file(parameters)
    result = runner.run(context)
    print(result)


if __name__ == "__main__":
    main()
