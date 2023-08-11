import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (
    AzureTextCompletion,
    OpenAITextCompletion,
    AzureChatCompletion,
    OpenAIChatCompletion
)
from semantic_kernel.orchestration.context_variables import ContextVariables

from dotenv import dotenv_values

_SUPPORTED_OPENAI_MDELS = ('text-davinci-003', 'gpt-3.5-turbo')


def get_kernel_from_env() -> sk.Kernel:
    """Get a kernel configured from the .env file."""

    # Determine whether we're using Azure or OpenAI.
    config = dotenv_values(".env")
    use_azure = (
        config.get("AZURE_OPENAI_DEPLOYMENT_NAME") is not None and
        config.get("AZURE_OPENAI_API_KEY") is not None and
        config.get("AZURE_OPENAI_ENDPOINT") is not None
    )
    if not use_azure:
        assert (
            config.get("OPENAI_API_KEY") is not None and
            config.get("OPENAI_ORG_ID") is not None and
            config.get("OPENAI_MODEL_ID") is not None
        ), ".env does not contain Azure or OpenAI settings."

    kernel = sk.Kernel()
    if use_azure:
        print(f"Initializing AZURE_OPENAI_DEPLOYMENT_NAME={config['AZURE_OPENAI_DEPLOYMENT_NAME']}")
        deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
        kernel.add_text_completion_service(
            "completion", AzureTextCompletion(deployment, endpoint, api_key)
        )
    else:
        model_id = config['OPENAI_MODEL_ID']
        assert model_id in _SUPPORTED_OPENAI_MDELS, f"OPENAI_MODEL_ID={model_id} is not supported. Supported models: {_SUPPORTED_OPENAI_MDELS}"
        print(f"Initializing OPENAI_MODEL_ID={model_id}")
        api_key, org_id = sk.openai_settings_from_dot_env()
        kernel.add_text_completion_service(
            "completion", OpenAITextCompletion(model_id, api_key, org_id)
        )
    return kernel
