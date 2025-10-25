from enum import Enum


class LocalModel(str, Enum):
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


class RemoteModel(str, Enum):
    """Remote LLMs."""

    GEMINI_2_0_FLASH_001 = "google/gemini-2.0-flash-001"
    GEMINI_2_5_FLASH = "google/gemini-2.5-flash"
    GPT_OSS_120B = "openai/gpt-oss-120b"
    GPT_OSS_20B = "openai/gpt-oss-20b"
    LLAMA_3_3_70B_INSTRUCT = "meta-llama/llama-3.3-70b-instruct"
    GPT_5_NANO = "openai/gpt-5-nano"
    NEMOTRON_NANO_9B_V2 = "nvidia/nemotron-nano-9b-v2"
    X_AI_GROK_4_FAST = "x-ai/grok-4-fast"
    X_AI_GROK_CODE_FAST_1 = "x-ai/grok-code-fast-1"
    Z_AI_GLM_4_5 = "z-ai/glm-4.5"
    QWEN3_30B_A3B = "qwen/qwen3-30b-a3b"
    QWEN3_NEXT_80B_A3B_INSTRUCT = "qwen/qwen3-next-80b-a3b-instruct"
    QWEN3_32B = "qwen/qwen3-32b"
