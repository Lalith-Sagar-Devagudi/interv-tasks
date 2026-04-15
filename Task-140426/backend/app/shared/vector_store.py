from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, ScrollRequest

from shared.config import (
    COLLECTION_NAME,
    QDRANT_API_KEY,
    QDRANT_URL,
    VECTOR_SIZE,
)
from shared.embedder import get_embeddings


class VectorStore:
    """Thin wrapper around Qdrant + OpenAI embeddings.

    Handles collection lifecycle (create-if-missing) so callers never need
    to think about Qdrant internals.
    """

    def __init__(self) -> None:
        self.collection_name = COLLECTION_NAME
        self.embeddings = get_embeddings()
        print(self.embeddings)
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _store(self) -> QdrantVectorStore:
        """Return a ready-to-use QdrantVectorStore bound to the collection."""
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings, 
        )

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            print(f"[vector_store] Created collection '{self.collection_name}'")

    # ── Public API ────────────────────────────────────────────────────────────

    def add_documents(self, documents: list[Document]) -> int:
        """Embed and upsert documents. Creates the collection if needed."""
        self._ensure_collection()
        self._store().add_documents(documents)
        return len(documents)

    def similarity_search_with_scores(
        self, query: str, top_k: int = 10
    ) -> list[tuple[Document, float]]:
        """Return (document, cosine_score) pairs sorted by relevance."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            return []
        return self._store().similarity_search_with_score(query, k=top_k)

    def keyword_search(self, keyword: str, limit: int = 5) -> list[dict]:
        """Scroll through all stored chunks and return those whose text contains
        the keyword (case-insensitive). Useful for debugging retrieval gaps."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            return []

        matches = []
        offset = None
        keyword_lower = keyword.lower()

        while True:
            result, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                with_vectors=False,
                limit=100,
                offset=offset,
            )
            for point in result:
                text = (point.payload or {}).get("page_content", "")
                if keyword_lower in text.lower():
                    matches.append({
                        "id": str(point.id),
                        "source": (point.payload or {}).get("metadata", {}).get("source", "unknown"),
                        "content": text,
                    })
                    if len(matches) >= limit:
                        return matches
            if next_offset is None:
                break
            offset = next_offset

        return matches

    def collection_exists(self) -> bool:
        existing = {c.name for c in self.client.get_collections().collections}
        return self.collection_name in existing
