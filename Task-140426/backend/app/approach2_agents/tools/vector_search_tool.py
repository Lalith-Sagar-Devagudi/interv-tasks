from langchain_core.tools import tool

from shared.vector_store import VectorStore

# Module-level store reference — injected at startup by the orchestrator
_store: VectorStore | None = None


def init_search_tool(store: VectorStore) -> None:
    """Inject the VectorStore instance before the tool is called."""
    global _store
    _store = store


@tool
def vector_search(query: str, top_k: int = 10) -> str:
    """Search the legal document vector store for relevant excerpts.

    Use this tool whenever you need evidence from the indexed documents
    to answer a legal question. Prefer specific, targeted queries over
    broad ones.

    Args:
        query: A targeted search query derived from the user question.
        top_k: Number of results to return (default 10, max 10).

    Returns:
        Numbered document excerpts with source filename and relevance score.
    """
    if _store is None:
        return "Error: vector store not initialised."

    results = _store.similarity_search_with_scores(query, top_k=min(top_k, 10))

    if not results:
        return "No relevant documents found for this query."

    parts = []
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(
            f"[Excerpt {i} | File: {source} | Score: {score:.4f}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(parts)
