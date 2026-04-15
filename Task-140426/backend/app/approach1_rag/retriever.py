from dataclasses import dataclass

from shared.config import RETRIEVAL_TOP_K
from shared.vector_store import VectorStore


@dataclass
class ScoredChunk:
    content: str
    score: float   # cosine similarity in [0, 1]
    source: str


class VectorRetriever:
    """Fetches the top-K most similar chunks from the vector store."""

    def __init__(self, store: VectorStore) -> None:
        self.store = store

    def retrieve(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> list[ScoredChunk]:
        print(f"[retriever] Querying Qdrant — top_k={top_k}  query={question!r}")
        results = self.store.similarity_search_with_scores(question, top_k=top_k)
        print(f"[retriever] Qdrant returned {len(results)} results")
        chunks = [
            ScoredChunk(
                content=doc.page_content,
                score=float(score),
                source=doc.metadata.get("source", "unknown"),
            )
            for doc, score in results
        ]
        return chunks
