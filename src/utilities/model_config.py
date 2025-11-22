from enum import StrEnum


class LocalModel(StrEnum):
    """
    Local LLMs.
    """

    MISTRAL_7B_INSTRUCT_V0_3_Q4_0 = "mistral:7b-instruct-v0.3-q4_0"
    LLAMA3_1_8B = "llama3.1:8b"
    LLAMA3_2_3B = "llama3.2:3b"
    QWEN3_1_7B = "qwen3-1.7b"
    QWEN3_4B = "qwen3:4b"
    QWEN3_4B_INSTRUCT_2507 = "qwen3-4b-instruct-2507"
    MXBAI_EMBED_LARGE = "mxbai-embed-large:latest"


class RemoteModel(StrEnum):
    """Remote LLMs."""

    DEEPSEEK_V3_2_EXP = "deepseek/deepseek-v3.2-exp"
    GEMINI_2_0_FLASH_001 = "google/gemini-2.0-flash-001"
    GEMINI_2_5_FLASH = "google/gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "google/gemini-2.5-flash-lite"
    GPT_OSS_120B = "openai/gpt-oss-120b"
    GPT_OSS_20B = "openai/gpt-oss-20b"
    GPT_5_NANO = "openai/gpt-5-nano"
    GROK_4_1_FAST = "x-ai/grok-4.1-fast"
    LLAMA_3_3_70B_INSTRUCT = "meta-llama/llama-3.3-70b-instruct"
    LLAMA_3_8B_INSTRUCT = "meta-llama/llama-3-8b-instruct"
    NEMOTRON_NANO_9B_V2 = "nvidia/nemotron-nano-9b-v2"
    QWEN3_30B_A3B = "qwen/qwen3-30b-a3b"
    QWEN3_NEXT_80B_A3B_INSTRUCT = "qwen/qwen3-next-80b-a3b-instruct"
    QWEN3_32B = "qwen/qwen3-32b"
    SAO10K_L3_LUNARIS_8B = "sao10k/l3-lunaris-8b"
    X_AI_GROK_4_FAST = "x-ai/grok-4-fast"
    X_AI_GROK_CODE_FAST_1 = "x-ai/grok-code-fast-1"
    Z_AI_GLM_4_5 = "z-ai/glm-4.5"


class RemoteEmbeddingModel(StrEnum):
    """Remote Embedding Models."""

    QWEN3_8B = "qwen/qwen3-embedding-8b"
    NVIDIA_NEMOTRON_NANO_12B_V2_VL_FREE = "nvidia/nemotron-nano-12b-v2-vl:free"
    OPENAI_TEXT_EMBEDDING_3_SMALL = "openai/text-embedding-3-small"
    THENLPER_GTE_BASE = "thenlper/gte-base"
    SENTENCE_TRANSFORMERS_PARAPHRASE_MINILM_L6_V2 = (
        "sentence-transformers/paraphrase-minilm-l6-v2"
    )
    SENTENCE_TRANSFORMERS_ALL_MINILM_L12_V2 = "sentence-transformers/all-minilm-l12-v2"
    SENTENCE_TRANSFORMERS_ALL_MPNET_BASE_V2 = "sentence-transformers/all-mpnet-base-v2"
