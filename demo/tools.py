from datetime import datetime

from langchain.tools import tool
from langchain_tavily import TavilySearch

from src.config import app_settings


@tool(response_format="content")
async def search_tool(query: str, max_chars: int = 500) -> str:
    """Perform a search using TavilySearch tool.

    Parameters:
    -----------
    query: str
        The search query.
    max_chars: int, default=1000
        The maximum number of characters per source to return from the search results.

    Returns:
    --------
    str
        The formatted search results.
    """
    separator: str = "\n\n"

    tavily_search = TavilySearch(
        api_key=app_settings.TAVILY_API_KEY.get_secret_value(),
        max_results=3,
        topic="general",
    )
    search_response = await tavily_search.ainvoke({"query": query})
    formatted_results: str = "\n\n".join(
        f"Title: {result['title']}\nContent: {result['content'][:max_chars]} [truncated]\nURL: {result['url']}{separator}"
        for result in search_response["results"]
    )
    return formatted_results


@tool(response_format="content")
def date_tool() -> str:
    """Get the current date and time.

    Returns:
    --------
    str
        The current date and time as a string.
    """

    return datetime.now().isoformat(timespec="seconds")
