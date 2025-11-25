from enum import StrEnum


class RetrieverMethodType(StrEnum):
    """The type of retrieval method to use for internal document search."""

    VECTOR_SEARCH = "vector_search"
    KEYWORD_SEARCH = "keyword_search"
    HYBRID_SEARCH = "hybrid_search"


class ToolsType(StrEnum):
    """The type of tool to use for each step."""

    VECTOR_STORE = "vector_store"
    WEB_SEARCH = "web_search"


class NextAction(StrEnum):
    """Tells the executor what to do after the current planning step."""

    CONTINUE = "continue"
    FINISH = "finish"


class FileFormatsType(StrEnum):
    """The type of file to process."""

    CSV = "csv"
    JSON = "json"
    TXT = "txt"
    PDF = "pdf"
