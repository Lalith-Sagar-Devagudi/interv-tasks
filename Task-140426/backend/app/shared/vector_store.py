from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

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
        """Return (document, cosine_score) pairs sorted by relevance.

        Uses hybrid retrieval: semantic search merged with keyword matches for
        any significant words in the query.  Keyword hits that don't appear in
        the semantic results are appended with a synthetic score of 0.50 so the
        LLM still sees them even when the embedding model underscores the chunk.
        """
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            return []

        semantic = self._store().similarity_search_with_score(query, k=top_k)

        # Extract candidate keywords: words ≥5 chars, not common stop-words
        _STOPWORDS = {"which", "about", "should", "according", "their", "that",
                      "with", "from", "what", "does", "have", "this", "will"}
        keywords = [
            w.strip("'\"?.,:;")
            for w in query.split()
            if len(w.strip("'\"?.,:;")) >= 6 and w.lower().strip("'\"?.,:;") not in _STOPWORDS
        ]

        if not keywords:
            return semantic

        # Collect IDs already in semantic results to avoid duplicates
        seen_ids = {str(doc.metadata.get("chunk_index", "")) + doc.metadata.get("source", "")
                    for doc, _ in semantic}

        extras: list[tuple[Document, float]] = []
        for kw in keywords[:3]:  # limit keyword passes to avoid over-fetching
            for match in self.keyword_search(kw, limit=5):
                uid = str(match.get("id", "")) + match.get("source", "")
                if uid in seen_ids:
                    continue
                seen_ids.add(uid)
                doc = Document(
                    page_content=match["content"],
                    metadata={"source": match["source"], "hybrid_keyword": kw},
                )
                extras.append((doc, 0.50))  # synthetic score signals keyword match

        return semantic + extras

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

    def reset_collection(self) -> None:
        """Drop the existing collection and recreate it with the current
        VECTOR_SIZE.  Call this before re-ingesting with a new embedding model."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name in existing:
            self.client.delete_collection(self.collection_name)
            print(f"[vector_store] Deleted collection '{self.collection_name}'")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"[vector_store] Recreated collection '{self.collection_name}' with size={VECTOR_SIZE}")

    def collection_exists(self) -> bool:
        existing = {c.name for c in self.client.get_collections().collections}
        return self.collection_name in existing

    def get_point_count(self) -> int:
        """Return the number of vectors currently stored in the collection."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            return 0
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0

    def get_all_chunk_texts(self) -> list[str]:
        """Scroll through every stored chunk and return raw text content.

        Returns one string per chunk (used for debug / keyword analysis).
        For TF-IDF topic extraction use ``get_texts_grouped_by_source()``
        which merges chunks back into per-document texts.
        """
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            return []

        texts: list[str] = []
        offset = None
        while True:
            result, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                with_vectors=False,
                limit=200,
                offset=offset,
            )
            for point in result:
                text = (point.payload or {}).get("page_content", "")
                if text.strip():
                    texts.append(text)
            if next_offset is None:
                break
            offset = next_offset

        return texts

    def get_texts_grouped_by_source(self) -> list[str]:
        """Return one merged text string per source document.

        Scrolls all chunks, groups them by their ``metadata.source`` value
        (the original PDF filename), and concatenates the chunk texts in
        chunk-index order.  This gives proper document-level TF-IDF weights
        so that long books (many chunks) don't dominate over short rulings.
        """
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            return []

        # source → list of (chunk_index, text)
        doc_chunks: dict[str, list[tuple[int, str]]] = {}
        offset = None
        while True:
            result, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                with_vectors=False,
                limit=200,
                offset=offset,
            )
            for point in result:
                payload  = point.payload or {}
                text     = payload.get("page_content", "")
                metadata = payload.get("metadata", {})
                source   = metadata.get("source", "unknown")
                idx      = metadata.get("chunk_index", 0)
                if text.strip():
                    doc_chunks.setdefault(source, []).append((idx, text))
            if next_offset is None:
                break
            offset = next_offset

        # Merge each source's chunks in order
        return [
            "\n\n".join(text for _, text in sorted(chunks))
            for chunks in doc_chunks.values()
        ]
