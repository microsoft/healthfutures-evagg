# Semantic Kernel - notes cache

These notes aren't useful in the current version of this repository, but will be revisited when we re-incorporate semantic kernel.

## Configuration

Execution of code within this repo requires execution of APIs hosted either by OpenAI or the Azure OpenAI Service. Accessing these APIs require configuration details that are specific to a given user and organization. **Do not share your api keys or other secrets; do not commit them to source control.**

Make sure you have an
[Open AI API Key](https://openai.com/api/) or
[Azure Open AI service key](https://learn.microsoft.com/azure/cognitive-services/openai/quickstart?pivots=rest-api)

Copy the `.env.example` file to a new file named `.env`. Then, copy those keys into the `.env` file:

Configuration of AZURE_OPENAI_ variables in your `.env` file will supersede configuration of `OPENAI...` variables, so in practice your configuration file will contain the following values if you're leveraging OpenAI APIs

```bash
OPENAI_API_KEY=""
OPENAI_ORG_ID=""
OPENAI_MODEL_ID=""
```

**Note: the key `OPENAI_MODEL_ID` is not directly used by Semantic Kernel in the way that the other two `OPENAI...` keys are. Instead this is where you will specify which model you're using (e.g., 'davinci-text-003'). Currently the scripts in this repo only leverage the completion API, so you'll need to specify a model here that supports the completion API.**

And it will alternatively contain the following values if you're levering the Azure OpenAI APIs

```bash
AZURE_OPENAI_DEPLOYMENT_NAME=""
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_API_KEY=""
```

## Running scripts

**Note: the current set of scripts are only configured to use the completion API (clients inherit from `TextCompletionClientBase`). Thus they cannot be configured to use GPT-4, whether provided by OAI or AOAI.**

To run a console application (i.e script) within Visual Studio Code, open the python file for that script and hit `F5`.
As configured in `launch.json` Visual Studio code will then run this script.

To run the console application from the terminal (`cwd` is the repository root), use the following commands:

```bash
export PYTHONPATH=$(pwd)/lib:$PYTHONPATH
python scripts/exec_sem_function.py
```

**Note: in both of these execution methods, the semantic function being executed and the json parameter file configuring that execution are implicitly specified. Review [scripts/exec_sem_function.py](exec_sem_function.py) or run `python scripts/exec_sem_function.py --help` for additional detail on how to modify this behavior.**

## Modifying prompts interactively

The Semantic Kernel extension provides a mechanism to modify prompts and execute them against a pre-configured LLM endpoint. Instructions for how to configure this are available in the [Semantic Kernel vscode tools documentation](https://learn.microsoft.com/en-us/semantic-kernel/vs-code-tools/).

Once configured, you can select a semantic function you wish to run (i.e., the `skprompt.txt` text file for that function) and execute it directly from within vscode. You will be prompted for arguments, I haven't yet figured out how to prespecify them or pick them up from a file.
