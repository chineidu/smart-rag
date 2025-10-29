from typing import Literal

from langchain.messages import RemoveMessage
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END

from demo import create_logger
from demo.prompts import no_summary_prompt, query_prompt, summary_prompt, sys_prompt
from demo.state import State
from demo.tools import date_and_time_tool, search_tool
from src.config import app_settings
from src.utilities.model_config import RemoteModel

logger = create_logger(name="nodes")
MAX_SUMMARY_TOKENS: int = 4096
MAX_MESSAGES: int = 50

llm = ChatOpenAI(
    api_key=app_settings.OPENROUTER_API_KEY.get_secret_value(),  # type: ignore
    base_url=app_settings.OPENROUTER_URL,  # type: ignore
    temperature=0.0,
    model=RemoteModel.GEMINI_2_0_FLASH_001,  # type: ignore
)
summarization_llm = ChatOpenAI(
    api_key=app_settings.OPENROUTER_API_KEY.get_secret_value(),  # type: ignore
    base_url=app_settings.OPENROUTER_URL,  # type: ignore
    temperature=0.0,
    model=RemoteModel.LLAMA_3_8B_INSTRUCT,  # type: ignore
).bind(max_tokens=MAX_SUMMARY_TOKENS)


async def llm_call_node(state: State) -> State:
    """Node to call the LLM with tools and conversation history."""
    summary: str = state.get("summary", "")
    sys_msg = SystemMessage(content=sys_prompt)

    if summary:
        summary_msg = SystemMessage(content=f"Summary of conversation:\n\n {summary}")
        # Summary + most recent messages
        msgs_with_summary = [summary_msg] + state["messages"]

    else:
        msgs_with_summary = state["messages"]

    _msg = query_prompt.format(query=state.get("query", ""))
    query = HumanMessage(content=_msg)
    llm_with_tools = llm.bind_tools(tools=[search_tool, date_and_time_tool])
    response = await llm_with_tools.ainvoke([sys_msg] + msgs_with_summary + [query])

    return State(
        query=state.get("query", ""),
        answer=response.content,  # type: ignore
        messages=[state.get("query", ""), response],  # type: ignore
        summary=summary,
    )


async def summarization_node(state: State) -> State:
    """Summarization node to condense the conversation history."""
    summary: str = state.get("summary", "")

    if summary:
        summary_msg: list[AnyMessage] = [
            HumanMessage(content=summary_prompt.format(summary=summary))
        ]
    else:
        summary_msg = [HumanMessage(content=no_summary_prompt)]

    response: AIMessage = await summarization_llm.ainvoke(
        state["messages"] + summary_msg
    )
    logger.info("Conversation history summarized.")

    # Delete ALL but the last 2 messages
    messages_to_remove = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]  # type: ignore

    return State(
        query=state.get("query", None),  # type: ignore
        answer=state.get("answer", None),  # type: ignore
        messages=messages_to_remove,  # type: ignore
        summary=response.content,  # type: ignore
    )


# ===============================================================
# =========================== EDGEs =============================
# ===============================================================
def should_summarize(state: State) -> Literal["summarize", END]:  # type: ignore
    """Edge to determine if summarization is needed."""
    if len(state["messages"]) > MAX_MESSAGES:
        return "summarize"

    return END
