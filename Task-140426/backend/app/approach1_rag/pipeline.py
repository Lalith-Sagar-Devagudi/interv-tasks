import asyncio
import time

from shared.config import LLM_INPUT_COST_PER_M, LLM_OUTPUT_COST_PER_M
from shared.vector_store import VectorStore
from approach1_rag.retriever import VectorRetriever
from approach1_rag.answer_generator import AnswerGenerator


class RAGPipeline:
    """Traditional RAG: retrieve → generate.

    Flow: user query → retrieve docs from VDB → (query + docs) + LLM → answer.
    Returns answer, confidence, latency, cost, and token counts.
    """

    def __init__(self, store: VectorStore) -> None:
        self.retriever = VectorRetriever(store)
        self.generator = AnswerGenerator()

    async def run(self, question: str) -> dict:
        start = time.perf_counter()

        print(f"[rag] Step 1/2 — Retrieving chunks for: {question!r}")
        chunks = self.retriever.retrieve(question)

        if not chunks:
            print("[rag] Step 1/2 — No chunks found. Returning early.")
            return {
                "answer": "No documents have been indexed yet, or no relevant content was found.",
                "confidence_score": 0.0,
                "latency_seconds": round(time.perf_counter() - start, 3),
                "cost_usd": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "approach": "traditional_rag",
                "sources": [],
            }

        print(f"[rag] Step 1/2 — Retrieved {len(chunks)} chunks:")
        for i, c in enumerate(chunks, 1):
            print(f"[rag]   {i}. source={c.source!r}  score={c.score:.4f}  preview={c.content[:80].replace(chr(10), ' ')!r}")

        context_parts = [
            f"[Source: {c.source} | Relevance: {c.score:.4f}]\n{c.content}"
            for c in chunks
        ]
        context = "\n\n---\n\n".join(context_parts)
        print(f"[rag] Step 1/2 — Context length: {len(context)} chars")

        print(f"[rag] Step 2/2 — Calling LLM ...")
        answer, usage = await asyncio.to_thread(self.generator.generate, question, context)
        print(f"[rag] Step 2/2 — Answer ({len(answer)} chars): {answer[:120]!r}{'...' if len(answer) > 120 else ''}")

        latency = round(time.perf_counter() - start, 3)
        input_tokens  = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cost = round(
            input_tokens  * LLM_INPUT_COST_PER_M  / 1_000_000 +
            output_tokens * LLM_OUTPUT_COST_PER_M / 1_000_000,
            8,
        )
        confidence = round(sum(c.score for c in chunks) / len(chunks), 4)
        sources = list({c.source for c in chunks})

        print(f"[rag] Done. latency={latency}s  cost=${cost:.8f}  tokens={input_tokens}+{output_tokens}  confidence={confidence}  sources={sources}")

        return {
            "answer": answer,
            "confidence_score": confidence,
            "latency_seconds": latency,
            "cost_usd": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "approach": "traditional_rag",
            "sources": sources,
        }
