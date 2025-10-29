import calendar
from datetime import datetime

from langchain.tools import tool
from langchain_tavily import TavilySearch

from src.config import app_settings


def day_name_tool(year: int, month: int, day: int) -> str:
    """Determine the name of the day for a given date.
    Parameters
    ----------
    year : int
        The year for which to generate the calendar (e.g., 2025).
    month : int
        The month for which to generate the calendar (1-12).
    day : int
        The day of the month (not used in calendar generation).

    Returns
    -------
    str
        The name of the day for the given date.
    """
    if not isinstance(year, int) or year < 1:
        raise ValueError("Year must be a positive integer.")
    if not isinstance(month, int):
        raise ValueError("Month must be an integer.")
    if not isinstance(day, int):
        raise ValueError("Day must be an integer.")
    if not (1 <= month <= 12):
        raise ValueError("Month must be between 1 and 12.")
    if not (1 <= day <= 31):
        raise ValueError("Day must be between 1 and 31.")

    return calendar.day_name[calendar.weekday(year, month, day)]


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
    search_response = await tavily_search.ainvoke({"query": query})  # type: ignore
    formatted_results: str = "\n\n".join(
        f"Title: {result['title']}\nContent: {result['content'][:max_chars]} [truncated]\nURL: {result['url']}{separator}"
        for result in search_response["results"]
    )
    return formatted_results


@tool(response_format="content")
def date_and_time_tool() -> str:
    """
    Get the current date, time, and day name as a string.

    Returns
    -------
    str
        The current date, time, and day name as a string.
    """
    raw_date: str = datetime.now().isoformat()
    date: str = raw_date.split("T")[0]
    time: str = raw_date.split("T")[-1]

    day_name: str = day_name_tool(
        year=int(date.split("-")[0]),
        month=int(date.split("-")[1]),
        day=int(date.split("-")[2]),
    )

    return f"Date: {date}\n\nTime: {time} (GMT+1)\n\nDay Name: {day_name}"
