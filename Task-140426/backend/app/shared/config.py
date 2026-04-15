import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")   # optional: direct OpenAI for embeddings
QDRANT_URL: str = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")

# ── OpenRouter ────────────────────────────────────────────────────────────────
# Used for LLM inference (/chat/completions).
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# ── OpenAI (direct) ───────────────────────────────────────────────────────────
# If OPENAI_API_KEY is set, embeddings go directly to api.openai.com instead
# of OpenRouter — useful when OpenRouter credits are exhausted.
OPENAI_BASE_URL: str = "https://api.openai.com/v1"

# ── Models ────────────────────────────────────────────────────────────────────
# Single model used by both approaches
LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek/deepseek-v3.2")

# Embeddings — text-embedding-3-small via OpenRouter /embeddings endpoint
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

# ── Qdrant ────────────────────────────────────────────────────────────────────
COLLECTION_NAME: str = "legal_docs"
VECTOR_SIZE: int = 1536  # text-embedding-3-small output dimension

# ── Chunking ──────────────────────────────────────────────────────────────────
# text-embedding-3-large supports up to 8191 tokens; 1500 chars ≈ 300–375 tokens
# which fits most full legal clauses and preserves argument context.
# 300-char overlap (20%) ensures clauses split at a boundary still appear
# complete in at least one neighbouring chunk.
CHUNK_SIZE: int = 1500
CHUNK_OVERLAP: int = 300

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = 20

# ── LLM generation limits ────────────────────────────────────────────────────
# Cap output tokens to avoid 402 "insufficient credits" errors from OpenRouter.
# Legal answers are concise; 1024 output tokens is plenty.
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ── Pricing (OpenRouter, USD per million tokens) ───────────────────────────────
# deepseek/deepseek-v3.2  — update if OpenRouter changes rates
LLM_INPUT_COST_PER_M: float  = float(os.getenv("LLM_INPUT_COST_PER_M",  "0.27"))
LLM_OUTPUT_COST_PER_M: float = float(os.getenv("LLM_OUTPUT_COST_PER_M", "1.10"))

