from enum import StrEnum

from diskcache.core import UNKNOWN


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


class ErrorCodeEnum(StrEnum):
    HTTP_ERROR = "http_error"
    INTERNAL_SERVER_ERROR = "internal_server_error"
    INVALID_INPUT = "invalid_input"
    MODEL_NOT_FOUND = "model_not_found"
    PREDICTION_ERROR = "prediction_error"
    RESOURCES_NOT_FOUND = "resources_not_found"


class ResourcesType(StrEnum):
    """The type of resource to use."""

    CACHE = "cache"
    DATABASE = "database"
    GRAPH = "graph"
    RATE_LIMITER = "rate_limiter"
    VECTOR_STORE = "vector_store"
    UNKNOWN = UNKNOWN


class SectionNamesType(StrEnum):
    """The type of filters available for retrieval searches."""

    ITEM_1 = "ITEM 1. BUSINESS"
    ITEM_1A = "ITEM 1A. RISK FACTORS"
    ITEM_1B = "ITEM 1B. UNRESOLVED STAFF COMMENTS"
    ITEM_2 = "ITEM 2. PROPERTIES"
    ITEM_3 = "ITEM 3. LEGAL PROCEEDINGS"
    ITEM_4 = "ITEM 4. MINE SAFETY DISCLOSURES"
    ITEM_5 = (
        "ITEM 5. MARKET FOR REGISTRANTS' COMMON EQUITY, RELATED STOCKHOLDER MATTERS AND ISSUER "
        "PURCHASES OF EQUITY SECURITIES"
    )
    ITEM_6 = "ITEM 6. [RESERVED]"
    ITEM_7 = "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS"
    ITEM_7A = "ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK"
    ITEM_8 = "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA"
    ITEM_9 = "ITEM 9. CHANGES IN AND DISAGREEMENTS WITH ACCOUNTANTS ON ACCOUNTING AND FINANCIAL DISCLOSURE"
    ITEM_9A = "ITEM 9A. CONTROLS AND PROCEDURES"
    ITEM_9C = (
        "ITEM 9C. DISCLOSURE REGARDING FOREIGN JURISDICTIONS THAT PREVENT INSPECTIONS"
    )
    ITEM_10 = "ITEM 10. DIRECTORS, EXECUTIVE OFFICERS AND CORPORATE GOVERNANCE"
    ITEM_11 = "ITEM 11. EXECUTIVE COMPENSATION"
    ITEM_12 = "ITEM 12. SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT AND RELATED STOCKHOLDER MATTERS"
    ITEM_13 = "ITEM 13. CERTAIN RELATIONSHIPS AND RELATED TRANSACTIONS, AND DIRECTOR INDEPENDENCE"
    ITEM_14 = "ITEM 14. PRINCIPAL ACCOUNTANT FEES AND SERVICES"
    ITEM_15 = "ITEM 15. EXHIBIT AND FINANCIAL STATEMENT SCHEDULES"
    ITEM_16 = "ITEM 16. FORM 10-K SUMMARY"
