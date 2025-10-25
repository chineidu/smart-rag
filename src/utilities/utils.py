from typing import Any

import instructor
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents.base import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langsmith import traceable
from openai import AsyncOpenAI
from pydantic import BaseModel

from src.config import app_settings
from src.utilities.model_config import RemoteModel

_async_client = AsyncOpenAI(
    api_key=app_settings.OPENROUTER_API_KEY.get_secret_value(),
    base_url=app_settings.OPENROUTER_URL,
)

aclient = instructor.from_openai(
    _async_client, mode=instructor.Mode.OPENROUTER_STRUCTURED_OUTPUTS
)

type PydanticModel = type[BaseModel]


def merge_dicts(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Merge two dictionaries, with new values updating existing ones."""
    if existing is None:
        return new
    # Update existing dict with new dict values
    return {**existing, **new}


@traceable
async def get_structured_output(
    messages: list[dict[str, Any]],
    model: RemoteModel,
    schema: PydanticModel,
) -> BaseModel:
    """
    Retrieves structured output from a chat completion model.

    Parameters
    ----------
    messages : list[dict[str, Any]]
        The list of messages to send to the model for the chat completion.
    model : RemoteModel
        The remote model to use for the chat completion (e.g., 'gpt-4o').
    schema : PydanticModel
        The Pydantic schema to enforce for the structured output.

    Returns
    -------
    BaseModel
        An instance of the provided Pydantic schema containing the structured output.

    Notes
    -----
    This is an asynchronous function that awaits the completion of the API call.
    """
    return await aclient.chat.completions.create(
        model=model,
        response_model=schema,
        messages=messages,
        max_retries=5,
    )


def convert_langchain_messages_to_dicts(
    messages: list[HumanMessage | SystemMessage | AIMessage],
) -> list[dict[str, str]]:
    """Convert LangChain messages to a list of dictionaries.

    Parameters
    ----------
    messages : list[HumanMessage | SystemMessage | AIMessage]
        List of LangChain message objects to convert.

    Returns
    -------
    list[dict[str, str]]
        List of dictionaries with 'role' and 'content' keys.
        Roles are mapped as follows:
        - HumanMessage -> "user"
        - SystemMessage -> "system"
        - AIMessage -> "assistant"

    """
    role_mapping: dict[str, str] = {
        "SystemMessage": "system",
        "HumanMessage": "user",
        "AIMessage": "assistant",
    }

    converted_messages: list[dict[str, str]] = []
    for msg in messages:
        message_type: str = msg.__class__.__name__
        # Default to "user" if unknown
        role: str = role_mapping.get(message_type, "user")
        converted_messages.append({"role": role, "content": msg.content})  # type: ignore

    return converted_messages


def load_pdf_doc(filepath: str, engine: str = "pypdfloader") -> list[Document]:
    """This is used to load a single document."""
    if engine == "pypdfloader":
        loader = PyPDFLoader(filepath)
    docs: list[Document] = loader.load()  # type: ignore
    return docs
