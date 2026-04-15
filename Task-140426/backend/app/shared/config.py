import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
QDRANT_URL: str = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")

# ── OpenRouter ────────────────────────────────────────────────────────────────
# Used for both LLM inference (/chat/completions) and embeddings (/embeddings).
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# ── Models ────────────────────────────────────────────────────────────────────
# Single model used by both approaches
LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek/deepseek-v3.2")

# Embeddings — Qwen3-Embedding-8B via OpenRouter /embeddings endpoint
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")

# ── Qdrant ────────────────────────────────────────────────────────────────────
COLLECTION_NAME: str = "legal_docs"
VECTOR_SIZE: int = 4096  # Qwen3-Embedding-8B output dimension

# ── Chunking ──────────────────────────────────────────────────────────────────
# Larger chunks preserve full legal clauses and cross-references
CHUNK_SIZE: int = 800     # ~600 tokens — fits one full legal clause
CHUNK_OVERLAP: int = 150  # overlap catches clauses split across boundaries

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = 20   # candidates pulled from vector DB

# ── LLM generation limits ────────────────────────────────────────────────────
# Cap output tokens to avoid 402 "insufficient credits" errors from OpenRouter.
# Legal answers are concise; 1024 output tokens is plenty.
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ── Pricing (OpenRouter, USD per million tokens) ───────────────────────────────
# deepseek/deepseek-v3.2  — update if OpenRouter changes rates
LLM_INPUT_COST_PER_M: float  = float(os.getenv("LLM_INPUT_COST_PER_M",  "0.27"))
LLM_OUTPUT_COST_PER_M: float = float(os.getenv("LLM_OUTPUT_COST_PER_M", "1.10"))

