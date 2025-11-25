from langchain_openai import ChatOpenAI

from src.config import app_config, app_settings
from src.utilities.model_config import RemoteModel

remote_llm = ChatOpenAI(
    api_key=app_settings.OPENROUTER_API_KEY.get_secret_value(),  # type: ignore
    base_url=app_settings.OPENROUTER_URL,
    temperature=app_config.llm_model_config.creative_model.temperature,
    model=app_config.llm_model_config.creative_model.model_name,
)

structured_output_model: RemoteModel = RemoteModel(
    app_config.llm_model_config.structured_output_model.model_name
)
