"""Embedding provider — text-embedding-3-small.

Routing priority:
  1. If OPENAI_API_KEY is set → call api.openai.com directly (no OpenRouter credits needed).
  2. Otherwise              → route through OpenRouter (requires embedding credits).

text-embedding-3-small produces 1536-dimensional vectors with strong
out-of-the-box retrieval quality — no instruction tuning required.
"""

from langchain_openai import OpenAIEmbeddings

from shared.config import (
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)


def get_embeddings() -> OpenAIEmbeddings:
    """Return embeddings, preferring a direct OpenAI key when available."""
    if OPENAI_API_KEY:
        # Strip the 'openai/' provider prefix — OpenAI's own API doesn't use it
        model = EMBEDDING_MODEL.removeprefix("openai/")
        return OpenAIEmbeddings(
            model=model,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
