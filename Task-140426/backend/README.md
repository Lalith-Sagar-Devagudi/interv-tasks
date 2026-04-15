# Legal QA API

A question-answering backend over legal PDF documents. Two independent approaches run in parallel on every `/ask` request so their answers, latency, cost, and confidence can be compared side-by-side.

---

## Quick Start

```bash
cd backend
uv run uvicorn app.app:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs` once running.

---

## Architecture Overview

### Shared Ingestion Pipeline (`POST /ingest`)

```
┌─────────────────────────────────────────────────────────────────┐
│                        POST /ingest                             │
│                    (upload PDF files)                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                ┌───────────▼───────────┐
                │      pdf_parser       │
                │  pymupdf4llm → text   │
                │  RecursiveTextSplitter│
                │  (1500 chars, 300     │
                │   overlap, legal      │
                │   separator hierarchy)│
                └───────────┬───────────┘
                            │  List[Document]
                ┌───────────▼───────────┐
                │      embedder         │
                │  text-embedding-3-    │
                │  small via OpenAI /   │
                │  OpenRouter           │
                └───────────┬───────────┘
                            │  1536-dim vectors
                ┌───────────▼───────────┐
                │    VectorStore        │
                │    (Qdrant Cloud)     │
                │  collection: legal_docs│
                └───────────────────────┘
```

---

### `POST /ask` — both approaches run in parallel

```
┌─────────────────────────────────────────────────────────────────┐
│                         POST /ask                               │
│               { "question": "..." }                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
            asyncio.gather — runs both in parallel
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
  ┌───────────────┐           ┌─────────────────────┐
  │  Approach 1   │           │     Approach 2      │
  │ Traditional   │           │   Multi-Agent RAG   │
  │    RAG        │           │   (Tool-Calling)    │
  └───────┬───────┘           └──────────┬──────────┘
          │                              │
          ▼                              ▼
  ┌───────────────┐           ┌─────────────────────┐
  │  VectorStore  │           │  Orchestrator LLM   │
  │ similarity    │           │                     │
  │ search top-20 │           │  Reads system prompt│
  │ (semantic +   │           │  → drives 3 tool    │
  │  keyword      │           │    calls in sequence│
  │  hybrid)      │           │  → answers itself   │
  └───────┬───────┘           └──────────┬──────────┘
          │                              │
          ▼                   ┌──────────▼──────────┐
  ┌───────────────┐           │  Tool 1             │
  │  AnswerGen    │           │  QueryRestructure   │
  │  DeepSeek V3  │           │  Agent (own LLM)    │
  │  via OpenRouter│          │                     │
  │               │           │  TF-IDF scope check │
  │  Answer only  │           │  → rewrite query or │
  │  from context │           │  → NOT_ANSWERABLE   │
  └───────┬───────┘           └──────────┬──────────┘
          │                              │
          │                   ┌──────────▼──────────┐
          │                   │  Tool 2             │
          │                   │  DocumentRetriever  │
          │                   │  (no LLM)           │
          │                   │                     │
          │                   │  Qdrant hybrid      │
          │                   │  search top-10      │
          │                   │  Stores full docs   │
          │                   │  internally         │
          │                   └──────────┬──────────┘
          │                              │
          │                   ┌──────────▼──────────┐
          │                   │  Tool 3             │
          │                   │  RelevanceValidator │
          │                   │  Agent (own LLM)    │
          │                   │                     │
          │                   │  Reads stored docs  │
          │                   │  → keeps only       │
          │                   │  relevant excerpts  │
          │                   │  → returns to       │
          │                   │  Orchestrator       │
          │                   └──────────┬──────────┘
          │                              │
          │                   ┌──────────▼──────────┐
          │                   │  Orchestrator LLM   │
          │                   │  (Step 4 — no tool) │
          │                   │                     │
          │                   │  Uses validated     │
          │                   │  excerpts already   │
          │                   │  in context to      │
          │                   │  write final answer │
          │                   └──────────┬──────────┘
          │                              │
          └──────────────┬───────────────┘
                         ▼
          ┌──────────────────────────────┐
          │   CombinedAnswerResponse     │
          │                             │
          │  traditional_rag: {         │
          │    answer, confidence_score, │
          │    latency_seconds, cost_usd,│
          │    input_tokens,            │
          │    output_tokens, sources   │
          │  }                          │
          │  agentic_rag: { same fields }│
          └──────────────────────────────┘
```

---

## Component Reference

### Shared Layer (`shared/`)

| Component | File | Role |
|---|---|---|
| `VectorStore` | `vector_store.py` | Qdrant wrapper. Creates collection on first ingest, exposes semantic + keyword hybrid search, document-level text export for TF-IDF |
| `pdf_parser` | `pdf_parser.py` | pymupdf4llm → markdown, then RecursiveCharacterTextSplitter with legal separator hierarchy (ARTICLE → SECTION → paragraph → sentence) |
| `embedder` | `embedder.py` | OpenAI `text-embedding-3-small` via direct API key or OpenRouter fallback |
| `TopicExtractor` | `topic_extractor.py` | Pure-Python TF-IDF over the full corpus. Extracts ~23 high-signal topic terms (bigrams + unigrams). Cache-aware: auto-refreshes when Qdrant point count changes after ingest |
| `config` | `config.py` | All env-var driven settings (models, keys, chunk sizes, pricing) |

---

### Approach 1 — Traditional RAG (`approach1_rag/`)

Single-pass deterministic pipeline. No LLM calls for retrieval.

```
user question
     │
     ▼
VectorRetriever.retrieve(question, top_k=20)
  └─ hybrid search: semantic (Qdrant cosine) + keyword scan
     │
     ▼
AnswerGenerator.generate(question, context)
  └─ DeepSeek V3 via OpenRouter
  └─ system: strict legal analyst, no hallucination
     │
     ▼
answer + avg_cosine_confidence + latency + cost
```

**LLM calls per request: 1**

---

### Approach 2 — Multi-Agent RAG (`approach2_agents/`)

An orchestrator LLM drives the pipeline through native tool-calling. Three specialist
sub-agents are registered as tools. The orchestrator's system prompt defines the
mandatory 4-step workflow — the LLM decides when each tool is called and generates
the final answer itself without delegating to a fourth agent.

```
Orchestrator LLM (reads system prompt)
     │
     ├─ tool call ──► QueryRestructureAgent (own LLM)
     │                 Scope check + query rewrite
     │                 → returns restructured query or NOT_ANSWERABLE
     │
     ├─ tool call ──► DocumentRetriever (no LLM — Qdrant only)
     │                 Hybrid vector + keyword search, top-10
     │                 → returns retrieval summary to orchestrator
     │                 → stores full excerpts internally for Tool 3
     │
     ├─ tool call ──► RelevanceValidatorAgent (own LLM)
     │                 Reads stored excerpts, keeps only relevant ones
     │                 → returns validated excerpts to orchestrator
     │
     └─ (no more tool calls)
          Orchestrator generates final answer from validated excerpts in context
```

**LLM instances:** 3 independent (`QueryRestructureAgent`, `RelevanceValidatorAgent`, `Orchestrator`)

**LLM calls per request:** 2–4 total
- 1 restructure (QueryRestructureAgent) — always
- 1 validate (RelevanceValidatorAgent) — skipped if no docs retrieved
- 1–2 orchestrator turns (more if the LLM requests multiple tool calls in sequence)
- Off-topic short-circuit: only 1 call (restructure returns NOT_ANSWERABLE, pipeline exits)

#### Orchestrator system prompt (abridged)

The orchestrator is given a strict 4-step workflow in its system prompt:

```
STEP 1 — RESTRUCTURE: call `restructure_query`
  ▸ If NOT_ANSWERABLE → stop, return fixed message

STEP 2 — RETRIEVE: call `retrieve_documents` with restructured query
  ▸ If no docs found → stop, return fixed message

STEP 3 — VALIDATE: call `validate_relevance` with the original question
  ▸ If no relevant excerpts → stop, return fixed message

STEP 4 — ANSWER: write the final answer yourself (no tool call)
  ▸ Ground every claim in the validated excerpts
  ▸ Cite article/clause numbers where present
```

#### Token accounting

Tokens are accumulated across all LLM instances and reported as a single total:

| Source | What is counted |
|---|---|
| `sub_input_tokens` / `sub_output_tokens` | QueryRestructureAgent + RelevanceValidatorAgent |
| `orch_input_tokens` / `orch_output_tokens` | All orchestrator LLM turns (tool-calling loop + final answer) |
| **Reported total** | sub + orch combined |

---

### TF-IDF Topic Extractor (`shared/topic_extractor.py`)

Prevents the QueryRestructureAgent from generating hallucinated queries for off-topic questions.

**How it works:**

```
VectorStore.get_texts_grouped_by_source()
  └─ scrolls all Qdrant chunks
  └─ merges chunks → one string per source PDF
  └─ document-level grouping ensures a 3-page ruling
     and a 300-page book each count as 1 document

extract_topics_from_texts(doc_texts)
  │
  ├─ tokenize: lowercase alpha tokens ≥ 5 chars, minus 3-layer stopword list
  │   Layer 1: English function words
  │   Layer 2: legal structural boilerplate (shall, pursuant, judgment…)
  │   Layer 3: corpus-specific noise (TOAEP editor names, form artifacts,
  │             month names, ICC Legal Tools cross-links)
  │
  ├─ per-doc TF-IDF (top-8 per document → coverage counter)
  │   Gives every document equal weight regardless of length
  │
  ├─ build unigram candidate pool (top-100 by coverage)
  │
  ├─ bigrams gated by candidate pool
  │   Both words must be independently meaningful
  │   Eliminates author names, boilerplate phrases
  │
  ├─ deduplicate: overlapping bigrams + suffix stemming
  │
  └─ return 8 bigrams + 15 unigrams (≈ 23 topics, ≈ 80 prompt tokens)

TopicExtractor (cache layer)
  └─ tracks Qdrant point count
  └─ auto-refreshes after each /ingest, no manual invalidation
```

**Sample output for this corpus (35 legal PDFs):**

```
international criminal, fraud corruption, african republic, trial chamber,
victims witnesses, witness testimony, investigation prosecution,
investigations financial, panel, procedure, investigative, government,
misconduct, disclosure, detention, reparations, accountability,
hypotheses, norwegian, statute, constitutional, referral, sexual
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Upload 1–10 PDF files → parse → embed → upsert into Qdrant |
| `POST` | `/ask` | Ask a question — both approaches run in parallel |
| `POST` | `/collection/reset` | Drop and recreate the Qdrant collection (use before re-ingesting after embedding model change) |
| `GET` | `/debug/collection-info` | Show Qdrant vector dimensions |
| `GET` | `/debug/keyword?q=...` | Keyword scan across all chunks |
| `GET` | `/debug/semantic?q=...` | High-k semantic search with scores |
| `GET` | `/health` | Liveness probe |

### `/ask` response shape

```json
{
  "question": "...",
  "traditional_rag": {
    "answer": "...",
    "confidence_score": 0.82,
    "latency_seconds": 1.4,
    "cost_usd": 0.00021,
    "input_tokens": 750,
    "output_tokens": 120,
    "sources": ["file.pdf"]
  },
  "agentic_rag": {
    "answer": "...",
    "confidence_score": 0.79,
    "latency_seconds": 6.2,
    "cost_usd": 0.00089,
    "input_tokens": 2800,
    "output_tokens": 410,
    "sources": ["file.pdf"]
  }
}
```

`input_tokens` and `output_tokens` in `agentic_rag` are the sum across all LLM instances
(orchestrator + QueryRestructureAgent + RelevanceValidatorAgent).

---

## Environment Variables

Create a `.env` file in `backend/`:

```env
# Required
OPENROUTER_API_KEY=sk-or-...
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-key

# Optional — direct OpenAI key skips OpenRouter for embeddings
OPENAI_API_KEY=sk-...

# Optional overrides
LLM_MODEL=deepseek/deepseek-v3.2
EMBEDDING_MODEL=openai/text-embedding-3-small
LLM_MAX_TOKENS=1024
LLM_INPUT_COST_PER_M=0.27
LLM_OUTPUT_COST_PER_M=1.10
```

**Embedding model and vector size must match.** If you change `EMBEDDING_MODEL`:
1. Call `POST /collection/reset` to recreate the Qdrant collection
2. Update `VECTOR_SIZE` in `shared/config.py` to match the new model's output dimension
3. Re-ingest all documents via `POST /ingest`

| Model | `VECTOR_SIZE` |
|---|---|
| `openai/text-embedding-3-small` | `1536` |
| `openai/text-embedding-3-large` | `3072` |

---

## Project Structure

```
backend/
├── app/
│   ├── app.py                              # FastAPI app, lifespan, all endpoints
│   ├── shared/
│   │   ├── config.py                       # All env-var settings
│   │   ├── embedder.py                     # OpenAI embeddings (direct or via OpenRouter)
│   │   ├── pdf_parser.py                   # PDF → chunks (pymupdf4llm + text splitter)
│   │   ├── vector_store.py                 # Qdrant wrapper (ingest, search, scroll)
│   │   └── topic_extractor.py             # TF-IDF corpus topic extraction + cache
│   ├── approach1_rag/
│   │   ├── pipeline.py                     # Orchestrates retrieve → generate
│   │   ├── retriever.py                    # Hybrid vector + keyword search
│   │   ├── answer_generator.py             # DeepSeek V3 answer synthesis
│   │   └── confidence_scorer.py           # Avg cosine similarity scorer
│   └── approach2_agents/
│       ├── orchestrator.py                 # Orchestrator LLM + tool-calling loop
│       └── tools/
│           ├── query_restructure_tool.py   # QueryRestructureAgent (own LLM)
│           └── relevance_validator_tool.py # RelevanceValidatorAgent (own LLM)
├── pyproject.toml
└── README.md
```

---

## Models Used

| Role | Model | Via | LLM instance |
|---|---|---|---|
| Answer generation (Approach 1) | `deepseek/deepseek-v3.2` | OpenRouter | `AnswerGenerator` |
| Query restructure (Approach 2 — Tool 1) | `deepseek/deepseek-v3.2` | OpenRouter | `QueryRestructureAgent` |
| Relevance validation (Approach 2 — Tool 3) | `deepseek/deepseek-v3.2` | OpenRouter | `RelevanceValidatorAgent` |
| Orchestration + final answer (Approach 2 — Step 4) | `deepseek/deepseek-v3.2` | OpenRouter | `LegalQAOrchestrator` |
| Embeddings | `text-embedding-3-small` (default) | OpenAI direct or OpenRouter | `embedder.get_embeddings()` |

All four LLM instances share the same model and pricing but are independent objects —
each can be swapped to a different model via env-var overrides if needed.

DeepSeek V3 context window: **64 K tokens**. The orchestrator system prompt uses ~400 tokens;
the restructure system prompt (corpus description + ~23 TF-IDF topics) uses ~530 tokens.

---

## Approach Comparison

| | Traditional RAG | Multi-Agent RAG |
|---|---|---|
| Architecture | Deterministic pipeline | Orchestrator LLM + 3 specialist sub-agents |
| Flow control | Code (hardcoded) | Orchestrator LLM (tool-calling loop) |
| LLM instances | 1 | 3 (orchestrator + restructure agent + validator agent) |
| LLM calls per query | 1 | 2–4 (varies by tool-calling turns) |
| Retrieval query | Raw user question | LLM-rewritten for VDB recall |
| Chunk filtering | None (top-20 passed to LLM) | RelevanceValidatorAgent filters before orchestrator answers |
| Off-topic handling | Answers "not found" after retrieval | QueryRestructureAgent detects before retrieval; orchestrator exits early |
| Final answer generated by | `AnswerGenerator` class | Orchestrator LLM itself (Step 4, no tool call) |
| Typical latency | ~1–4 s | ~5–12 s |
| Typical cost | Lower | Higher (3 LLM instances, 2–4 calls) |
| Answer quality | Good on simple factual queries | Better on complex/ambiguous queries |
