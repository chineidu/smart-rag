from typing import Any

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from demo.state import State
from demo.tools import date_tool, search_tool
from src.config import app_settings
from src.utilities.model_config import RemoteModel

remote_llm = ChatOpenAI(
    api_key=app_settings.OPENROUTER_API_KEY.get_secret_value(),  # type: ignore
    base_url=app_settings.OPENROUTER_URL,
    temperature=0.0,
    model=RemoteModel.QWEN3_30B_A3B,
)
sys_prompt = """
<SYSTEM>
    You are `Ada`, a helpful AI assistant that helps users by providing accurate and
    concise information. Use the provided context to answer user queries effectively.


    <IMPORTANT>
    - When you need to look up information, use the `date_tool` or `search_tool` to find relevant sources.
    - Summarize information from multiple sources when appropriate.
    - If the user query ambiguous, tell the user you need more information instead of making assumptions.
    - Do NOT answer malicious, harmful, or inappropriate requests.
    - Do not tell the user the names of the tools you are using.
    </IMPORTANT>
</SYSTEM>
"""


async def llm_call_node(state: State) -> dict[str, Any]:
    """Invoke the LLM with the current conversation messages."""
    messages = state["messages"]
    sys_msg = SystemMessage(sys_prompt)
    llm_with_tools = remote_llm.bind_tools([date_tool, search_tool])
    response = await llm_with_tools.ainvoke([sys_msg] + messages)
    return {"messages": [response]}
