"""Agentic RAG orchestrator using LangGraph.

Graph topology
──────────────
  START
    │
    ▼
  restructure_query   (Tool 1) — LLM rewrites query for better VDB recall
    │
    ▼
  retrieve_docs       (Tool 2) — semantic search against vector DB
    │
    ▼
  validate_relevance  (Tool 3) — LLM judges which chunks truly answer the question
    │
    ▼
  generate_answer     — LLM produces final answer from (original question + validated docs)
    │
    ▼
   END

All graph nodes are synchronous.  The entire graph is run via
asyncio.to_thread(graph.invoke, ...) so it executes in a worker thread
without blocking the event loop, allowing the RAG pipeline to run truly
in parallel via asyncio.gather.

Token usage is accumulated across all three LLM nodes so cost reflects
the full agentic pipeline.
"""

import time
import traceback
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from approach2_agents.tools.query_restructure_tool import (
    _do_restructure,
    init_restructure_tool,
)
from approach2_agents.tools.relevance_validator_tool import (
    _do_validate,
    init_validator_tool,
)
from approach2_agents.tools.vector_search_tool import init_search_tool
from shared.config import (
    LLM_INPUT_COST_PER_M,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_OUTPUT_COST_PER_M,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)
from shared.vector_store import VectorStore


# ── State schema ──────────────────────────────────────────────────────────────

class QAState(TypedDict):
    question: str             # original user question — never mutated
    restructured_query: str   # query rewritten by Tool 1
    retrieved_docs: str       # formatted excerpts from Tool 2
    validated_docs: str       # filtered excerpts from Tool 3
    sources: list[str]        # unique source filenames from retrieval
    answer: str               # final LLM answer
    confidence_score: float   # avg cosine similarity of retrieved chunks
    input_tokens: int         # accumulated across all LLM calls
    output_tokens: int        # accumulated across all LLM calls


# ── Orchestrator ──────────────────────────────────────────────────────────────

class LegalQAOrchestrator:

    def __init__(self, store: VectorStore) -> None:
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            max_tokens=LLM_MAX_TOKENS,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        init_search_tool(store)
        init_restructure_tool(self.llm)
        init_validator_tool(self.llm)
        self._store = store
        self.graph = self._build_graph()

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(QAState)

        g.add_node("restructure_query",   self._restructure_node)
        g.add_node("retrieve_docs",       self._retrieve_node)
        g.add_node("validate_relevance",  self._validate_node)
        g.add_node("generate_answer",     self._generate_node)

        g.add_edge(START,                "restructure_query")
        g.add_edge("restructure_query",  "retrieve_docs")
        g.add_edge("retrieve_docs",      "validate_relevance")
        g.add_edge("validate_relevance", "generate_answer")
        g.add_edge("generate_answer",    END)

        return g.compile()

    # ── Nodes (sync — run inside a worker thread via asyncio.to_thread) ───────

    def _restructure_node(self, state: QAState) -> dict:
        """Tool 1 — Rewrite the user question into a targeted VDB search query."""
        print(f"[agent:restructure] Original question: {state['question']!r}")
        restructured, usage = _do_restructure(self.llm, state["question"])
        print(f"[agent:restructure] Restructured query: {restructured!r}  usage={usage}")
        return {
            "restructured_query": restructured,
            "input_tokens":  state.get("input_tokens",  0) + usage.get("input_tokens",  0),
            "output_tokens": state.get("output_tokens", 0) + usage.get("output_tokens", 0),
        }

    def _retrieve_node(self, state: QAState) -> dict:
        """Tool 2 — Retrieve document excerpts using the restructured query."""
        query = state["restructured_query"]
        print(f"[agent:retrieve] Querying VDB with: {query!r}")

        results = self._store.similarity_search_with_scores(query, top_k=10)
        print(f"[agent:retrieve] Qdrant returned {len(results)} results")

        if not results:
            print("[agent:retrieve] No results found")
            return {"retrieved_docs": "", "sources": [], "confidence_score": 0.0}

        parts, sources, scores = [], [], []
        for i, (doc, score) in enumerate(results, 1):
            src = doc.metadata.get("source", "unknown")
            print(f"[agent:retrieve]   {i}. source={src!r}  score={score:.4f}  preview={doc.page_content[:80].replace(chr(10), ' ')!r}")
            parts.append(f"[Excerpt {i} | File: {src} | Score: {score:.4f}]\n{doc.page_content}")
            scores.append(float(score))
            if src not in sources:
                sources.append(src)

        confidence = round(sum(scores) / len(scores), 4)
        return {
            "retrieved_docs": "\n\n---\n\n".join(parts),
            "sources": sources,
            "confidence_score": confidence,
        }

    def _validate_node(self, state: QAState) -> dict:
        """Tool 3 — Filter retrieved excerpts down to only those relevant to the question."""
        print(f"[agent:validate] Validating {len(state.get('retrieved_docs', ''))} chars of retrieved docs ...")

        if not state.get("retrieved_docs"):
            print("[agent:validate] No docs to validate")
            return {"validated_docs": ""}

        validated, usage = _do_validate(self.llm, state["question"], state["retrieved_docs"])
        print(f"[agent:validate] Validated docs: {len(validated)} chars  usage={usage}")
        return {
            "validated_docs": validated,
            "input_tokens":  state.get("input_tokens",  0) + usage.get("input_tokens",  0),
            "output_tokens": state.get("output_tokens", 0) + usage.get("output_tokens", 0),
        }

    def _generate_node(self, state: QAState) -> dict:
        """Final LLM call — answer the original question using validated excerpts."""
        print(f"[agent:generate] Generating answer with model={LLM_MODEL!r} ...")

        context = state.get("validated_docs") or state.get("retrieved_docs", "")
        if not context:
            print("[agent:generate] No context available")
            return {"answer": "The provided documents do not contain information to answer this question."}

        system_content = """\
You are a precise legal document analyst.
Answer the user's question based strictly on the provided document excerpts.

Rules:
- Use only information present in the excerpts — do not hallucinate.
- If the answer cannot be found, say: "The provided documents do not contain sufficient information to answer this question."
- Be concise; cite clause numbers or section headings when available."""
        human_content = f"Document excerpts:\n{context}\n\nQuestion: {state['question']}\n\nAnswer:"

        print("[agent:generate] ── FULL PROMPT ──────────────────────────────")
        print(f"[agent:generate] SYSTEM:\n{system_content}")
        print(f"[agent:generate] HUMAN ({len(human_content)} chars):\n{human_content}")
        print("[agent:generate] ── END PROMPT ──────────────────────────────")

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]

        response = self.llm.invoke(messages)
        usage = response.usage_metadata or {}
        answer = response.content.strip()
        print(f"[agent:generate] Answer ({len(answer)} chars): {answer[:120]!r}{'...' if len(answer) > 120 else ''}  usage={usage}")

        return {
            "answer": answer,
            "input_tokens":  state.get("input_tokens",  0) + usage.get("input_tokens",  0),
            "output_tokens": state.get("output_tokens", 0) + usage.get("output_tokens", 0),
        }

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(self, question: str) -> dict:
        print(f"[agent] ▶ Starting agentic RAG for: {question!r}")
        start = time.perf_counter()

        initial: QAState = {
            "question": question,
            "restructured_query": "",
            "retrieved_docs": "",
            "validated_docs": "",
            "sources": [],
            "answer": "",
            "confidence_score": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        # Use ainvoke so LangGraph manages thread-pool execution of sync nodes
        # internally — avoids asyncio event-loop conflicts that arise when
        # wrapping graph.invoke() in asyncio.to_thread().
        try:
            final = await self.graph.ainvoke(initial)
        except Exception:
            traceback.print_exc()
            raise

        latency = round(time.perf_counter() - start, 3)
        input_tokens  = final.get("input_tokens",  0)
        output_tokens = final.get("output_tokens", 0)
        cost = round(
            input_tokens  * LLM_INPUT_COST_PER_M  / 1_000_000 +
            output_tokens * LLM_OUTPUT_COST_PER_M / 1_000_000,
            8,
        )

        print(f"[agent] ◀ Done — latency={latency}s  cost=${cost:.8f}  tokens={input_tokens}+{output_tokens}  confidence={final['confidence_score']}  sources={final.get('sources', [])}")

        return {
            "answer": final["answer"],
            "confidence_score": final["confidence_score"],
            "latency_seconds": latency,
            "cost_usd": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "approach": "agentic_rag",
            "sources": final.get("sources", []),
        }
