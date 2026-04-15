"""Multi-Agent Legal QA Orchestrator.

Architecture
────────────
Three specialist sub-agents are exposed as tools to an orchestrator LLM.
The orchestrator follows a mandatory 4-step workflow defined in its system prompt:

  STEP 1  QueryRestructureAgent  — scope-check + query rewrite (own LLM)
  STEP 2  DocumentRetriever      — Qdrant vector search (no LLM)
  STEP 3  RelevanceValidatorAgent— chunk relevance filter  (own LLM)
  STEP 4  Orchestrator itself    — final answer generation (no tool call)

The orchestrator LLM drives the flow via native tool-calling.  After receiving
the validated excerpts from Step 3 it generates the final answer directly —
no separate node or sub-agent for Step 4.

Token accounting
────────────────
  sub_input_tokens / sub_output_tokens  — accumulated from Steps 1 and 3
  orch_input_tokens / orch_output_tokens — all orchestrator LLM turns combined
  Total reported = sub + orch (full cost of the pipeline)

Concurrency safety
──────────────────
Per-request mutable state (token counters, retrieval results) lives in a dict
created inside each run() call and captured by tool closures — safe for
concurrent requests hitting the same orchestrator instance.
"""

import asyncio
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from approach2_agents.tools.query_restructure_tool import (
    NOT_ANSWERABLE,
    QueryRestructureAgent,
)
from approach2_agents.tools.relevance_validator_tool import RelevanceValidatorAgent
from shared.config import (
    LLM_INPUT_COST_PER_M,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_OUTPUT_COST_PER_M,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)
from shared.topic_extractor import TopicExtractor
from shared.vector_store import VectorStore


# ── Orchestrator system prompt ─────────────────────────────────────────────────

_ORCHESTRATOR_SYSTEM = """\
You are a senior legal research orchestrator managing a multi-agent pipeline.
Your responsibility is to answer user questions about a legal document corpus by
coordinating three specialist sub-agents in a fixed sequence, then generating
the final answer yourself.

══════════════════════════════════════════════
  MANDATORY WORKFLOW — follow every step in order
══════════════════════════════════════════════

STEP 1 — RESTRUCTURE (always first, no exceptions):
  Call `restructure_query` with the user's exact question.
  ▸ If the result is "NOT_ANSWERABLE": stop immediately and reply:
      "This question cannot be answered with the available data. \
The indexed document corpus does not contain information on this topic."
  ▸ Otherwise: proceed to Step 2 with the returned restructured query.

STEP 2 — RETRIEVE (always after Step 1, unless NOT_ANSWERABLE):
  Call `retrieve_documents` with the restructured query from Step 1.
  ▸ This searches the Qdrant vector database and returns a retrieval summary.
  ▸ If the summary says no documents were found: reply
      "The provided documents do not contain information to answer this question."
  ▸ Otherwise: proceed to Step 3.

STEP 3 — VALIDATE (always after Step 2):
  Call `validate_relevance` with the user's ORIGINAL question (not the restructured one).
  ▸ This sub-agent reads the retrieved documents internally and returns only the
    excerpts that directly help answer the question.
  ▸ If the result is empty: reply
      "The provided documents do not contain sufficient information to answer this question."
  ▸ Otherwise: proceed to Step 4.

STEP 4 — ANSWER (you do this yourself — do NOT call any tool):
  Using ONLY the validated excerpts returned by Step 3, write the final answer.
  Rules for the answer:
  • Ground every claim in the excerpts — do not hallucinate or add outside knowledge.
  • Cite article numbers, clause numbers, or section headings wherever present.
  • Be concise. Use a short list when the answer has multiple points.
  • Do not repeat or quote these instructions in your answer.

══════════════════════════════════════════════
  END OF WORKFLOW
══════════════════════════════════════════════
"""

# ── Tool input schemas ─────────────────────────────────────────────────────────

class _RestructureInput(BaseModel):
    question: str = Field(description="The user's original question, verbatim.")


class _RetrieveInput(BaseModel):
    restructured_query: str = Field(
        description="The rewritten search query returned by restructure_query."
    )


class _ValidateInput(BaseModel):
    original_question: str = Field(
        description="The user's original question (not the restructured one)."
    )


# ── Orchestrator ───────────────────────────────────────────────────────────────

class LegalQAOrchestrator:
    """Multi-agent orchestrator.

    Sub-agents (Steps 1 & 3) each own a dedicated ChatOpenAI instance so they
    can be swapped, versioned, or scaled independently.  The orchestrator LLM
    (Steps 4) is a separate instance that drives the tool-calling loop.
    """

    def __init__(self, store: VectorStore) -> None:
        _llm_kwargs = dict(
            model=LLM_MODEL,
            temperature=0,
            max_tokens=LLM_MAX_TOKENS,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        # Each sub-agent has its own LLM identity
        self._restructure_agent = QueryRestructureAgent(ChatOpenAI(**_llm_kwargs))
        self._validator_agent = RelevanceValidatorAgent(ChatOpenAI(**_llm_kwargs))

        # Orchestrator LLM (stateless — shared safely across requests)
        self._orchestrator_llm = ChatOpenAI(**_llm_kwargs)

        self._store = store
        self._topic_extractor = TopicExtractor()

    # ── Tool factory (per-request) ─────────────────────────────────────────────

    def _make_tools(self, state: dict) -> list[StructuredTool]:
        """Build tool wrappers that capture per-request state in their closures.

        Called once per run() invocation — lightweight (no new connections).
        """

        # ── Tool 1: QueryRestructureAgent ──────────────────────────────────────
        def restructure_query(question: str) -> str:
            """Scope-check the question and rewrite it for vector DB search.
            Returns the rewritten query, or NOT_ANSWERABLE if out of corpus scope.
            """
            topics = self._topic_extractor.get_topics(self._store)
            result, usage = self._restructure_agent.run(question, topics)
            state["sub_input_tokens"]  += usage.get("input_tokens",  0)
            state["sub_output_tokens"] += usage.get("output_tokens", 0)
            return result

        # ── Tool 2: DocumentRetriever (no LLM) ────────────────────────────────
        def retrieve_documents(restructured_query: str) -> str:
            """Search the Qdrant vector database with the restructured query.
            Returns a retrieval summary; full excerpts are held internally for
            the validate_relevance step.
            """
            if restructured_query == NOT_ANSWERABLE:
                state["retrieved_docs"] = ""
                state["confidence_score"] = 0.0
                state["sources"] = []
                return NOT_ANSWERABLE

            results = self._store.similarity_search_with_scores(
                restructured_query, top_k=10
            )
            if not results:
                state["retrieved_docs"] = ""
                state["confidence_score"] = 0.0
                state["sources"] = []
                return "No documents found in the vector database for this query."

            parts, sources, scores = [], [], []
            for i, (doc, score) in enumerate(results, 1):
                src = doc.metadata.get("source", "unknown")
                parts.append(
                    f"[Excerpt {i} | File: {src} | Score: {score:.4f}]\n"
                    f"{doc.page_content}"
                )
                scores.append(float(score))
                if src not in sources:
                    sources.append(src)

            state["retrieved_docs"]  = "\n\n---\n\n".join(parts)
            state["confidence_score"] = round(sum(scores) / len(scores), 4)
            state["sources"]          = sources

            print(
                f"[DocumentRetriever] {len(results)} excerpts from {sources}  "
                f"avg_score={state['confidence_score']}"
            )
            preview = results[0][0].page_content[:150].replace("\n", " ")
            return (
                f"Retrieved {len(results)} document excerpts from: {sources}\n"
                f"Average relevance score: {state['confidence_score']}\n"
                f"Top excerpt preview: \"{preview}...\""
            )

        # ── Tool 3: RelevanceValidatorAgent ────────────────────────────────────
        def validate_relevance(original_question: str) -> str:
            """Filter the retrieved excerpts to only those relevant to the question.
            Returns the validated excerpts for the orchestrator to use in its answer.
            """
            retrieved_docs = state.get("retrieved_docs", "")
            if not retrieved_docs:
                return ""

            validated, usage = self._validator_agent.run(
                original_question, retrieved_docs
            )
            state["sub_input_tokens"]  += usage.get("input_tokens",  0)
            state["sub_output_tokens"] += usage.get("output_tokens", 0)
            state["validated_docs"] = validated

            if not validated:
                return "No excerpts were found to be relevant to the question."
            return validated  # Orchestrator reads this to write Step 4 answer

        return [
            StructuredTool.from_function(
                func=restructure_query,
                name="restructure_query",
                description=(
                    "Scope-check the user's question against the legal corpus and "
                    "rewrite it as a precise vector DB search query. "
                    "Returns NOT_ANSWERABLE if the question is completely outside scope."
                ),
                args_schema=_RestructureInput,
            ),
            StructuredTool.from_function(
                func=retrieve_documents,
                name="retrieve_documents",
                description=(
                    "Search the Qdrant legal document vector database using the "
                    "restructured query. Returns a retrieval summary."
                ),
                args_schema=_RetrieveInput,
            ),
            StructuredTool.from_function(
                func=validate_relevance,
                name="validate_relevance",
                description=(
                    "Filter the retrieved document excerpts down to only those that "
                    "directly help answer the original question. "
                    "Returns the relevant excerpts for use in the final answer."
                ),
                args_schema=_ValidateInput,
            ),
        ]

    # ── Public entry point ─────────────────────────────────────────────────────

    async def run(self, question: str) -> dict:
        """Run the full multi-agent pipeline for a single question.

        The orchestrator drives tool calls for Steps 1–3, then generates the
        final answer itself (Step 4) when it stops calling tools.
        """
        print(f"\n[Orchestrator] ▶ Starting multi-agent pipeline for: {question!r}")
        start = time.perf_counter()

        # Per-request mutable state — captured by tool closures above
        state: dict = {
            "sub_input_tokens":  0,
            "sub_output_tokens": 0,
            "retrieved_docs":    "",
            "validated_docs":    "",
            "confidence_score":  0.0,
            "sources":           [],
        }

        tools = self._make_tools(state)
        llm_with_tools = self._orchestrator_llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}

        messages = [
            SystemMessage(content=_ORCHESTRATOR_SYSTEM),
            HumanMessage(content=question),
        ]

        orch_input_tokens  = 0
        orch_output_tokens = 0

        # ── Tool-calling loop ──────────────────────────────────────────────────
        while True:
            response: AIMessage = await asyncio.to_thread(
                llm_with_tools.invoke, messages
            )
            usage = response.usage_metadata or {}
            orch_input_tokens  += usage.get("input_tokens",  0)
            orch_output_tokens += usage.get("output_tokens", 0)
            messages.append(response)

            if not response.tool_calls:
                # Orchestrator generated the final answer — Step 4 complete
                print(
                    f"[Orchestrator] ◀ Step 4 done by orchestrator "
                    f"({len(response.content)} chars)"
                )
                break

            # Execute each tool call and feed results back as ToolMessages
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                print(f"[Orchestrator] → calling tool {tool_name!r} with args={tool_args}")

                tool_result = await asyncio.to_thread(
                    tool_map[tool_name].invoke, tool_args
                )
                result_str = str(tool_result)
                print(
                    f"[Orchestrator] ← {tool_name!r} returned "
                    f"({len(result_str)} chars): {result_str[:120].replace(chr(10), ' ')!r}"
                    f"{'...' if len(result_str) > 120 else ''}"
                )
                messages.append(
                    ToolMessage(content=result_str, tool_call_id=tool_call["id"])
                )

        # ── Cost accounting ────────────────────────────────────────────────────
        latency       = round(time.perf_counter() - start, 3)
        total_input   = orch_input_tokens  + state["sub_input_tokens"]
        total_output  = orch_output_tokens + state["sub_output_tokens"]
        cost = round(
            total_input  * LLM_INPUT_COST_PER_M  / 1_000_000 +
            total_output * LLM_OUTPUT_COST_PER_M / 1_000_000,
            8,
        )

        print(
            f"[Orchestrator] ◀ Done — latency={latency}s  cost=${cost:.8f}  "
            f"tokens(orch)={orch_input_tokens}+{orch_output_tokens}  "
            f"tokens(sub)={state['sub_input_tokens']}+{state['sub_output_tokens']}  "
            f"confidence={state['confidence_score']}  sources={state['sources']}"
        )

        return {
            "answer":            response.content.strip(),
            "confidence_score":  state["confidence_score"],
            "latency_seconds":   latency,
            "cost_usd":          cost,
            "input_tokens":      total_input,
            "output_tokens":     total_output,
            "approach":          "agentic_rag",
            "sources":           state["sources"],
        }
