from approach1_rag.retriever import ScoredChunk
from shared.config import CONTEXT_TOP_N


class ConfidenceScorer:
    """Derives a confidence score from Qdrant's cosine-similarity scores.

    Formula: average cosine similarity of the top-N retrieved chunks.

    Cosine similarity from Qdrant is already in [0, 1], where 1 = identical
    vectors. Averaging the top-N gives a reliable signal of how well the
    corpus covers the query — no additional model inference required.
    """

    def compute(self, chunks: list[ScoredChunk], top_n: int = CONTEXT_TOP_N) -> float:
        if not chunks:
            return 0.0
        top_scores = [c.score for c in chunks[:top_n]]
        return round(sum(top_scores) / len(top_scores), 4)
