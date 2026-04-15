"""Embedding provider — Qwen3-Embedding-8B via OpenRouter.

OpenRouter exposes an /embeddings endpoint (OpenAI-compatible format) for
supported embedding models. We reuse langchain-openai's OpenAIEmbeddings,
pointing its base_url at OpenRouter and swapping in the OpenRouter API key.

Qwen3-Embedding-8B is an instruction-tuned model that uses ASYMMETRIC retrieval:
  - Documents: embedded WITHOUT an instruction prefix (as indexed).
  - Queries:   embedded WITH an instruction prefix so the query vector aligns
               with the document vectors in the retrieval space.

Adding the prefix only to embed_query() means no re-indexing is needed.
"""

from langchain_openai import OpenAIEmbeddings

from shared.config import EMBEDDING_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL

_QUERY_INSTRUCTION = (
    "Instruct: Given a user question about legal documents, "
    "retrieve relevant legal document excerpts that answer the question\nQuery: "
)


class _Qwen3Embeddings(OpenAIEmbeddings):
    """OpenAIEmbeddings subclass that prepends the Qwen3 retrieval instruction
    to queries only, leaving document embeddings unchanged."""

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(_QUERY_INSTRUCTION + text)

    async def aembed_query(self, text: str) -> list[float]:
        return await super().aembed_query(_QUERY_INSTRUCTION + text)


def get_embeddings() -> _Qwen3Embeddings:
    """Return embeddings backed by Qwen3-Embedding-8B via OpenRouter."""
    return _Qwen3Embeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
