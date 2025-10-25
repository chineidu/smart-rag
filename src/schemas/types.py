from enum import Enum


# ==================================================================
# ============================= TYPES ==============================
# ==================================================================
class YesOrNo(str, Enum):
    YES = "yes"
    NO = "no"


class DataSource(str, Enum):
    VECTORSTORE = "vectorstore"
    WEBSEARCH = "websearch"


class VectorSearchType(str, Enum):
    FOOTBALL = "football news"  # "(arsenal news | chelsea news | liverpool news)"
    AI = "ai news"  # "(ai news | ai browser | nvidia | openai| tech in china)"
