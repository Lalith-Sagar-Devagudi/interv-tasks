from approach1_rag.retriever import ScoredChunk
from shared.config import CONTEXT_TOP_N


class ContextBuilder:
    """Assembles the top-N scored chunks into a single context string."""

    def build(self, chunks: list[ScoredChunk], top_n: int = CONTEXT_TOP_N) -> str:
        top = chunks[:top_n]
        parts = [
            f"[Source: {c.source} | Relevance: {c.score:.4f}]\n{c.content}"
            for c in top
        ]
        return "\n\n---\n\n".join(parts)
