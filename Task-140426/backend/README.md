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
                │  (1500 chars, 200     │
                │   overlap, legal      │
                │   separator hierarchy)│
                └───────────┬───────────┘
                            │  List[Document]
                ┌───────────▼───────────┐
                │      embedder         │
                │  text-embedding-3-*   │
                │  via OpenAI / OpenRouter│
                └───────────┬───────────┘
                            │  1536-dim vectors
                ┌───────────▼───────────┐
                │    VectorStore        │
                │    (Qdrant Cloud)     │
                │  collection: legal_docs│
                └───────────────────────┘
```

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
  │ Traditional   │           │   Agentic RAG        │
  │    RAG        │           │   (LangGraph)        │
  └───────┬───────┘           └──────────┬──────────┘
          │                              │
          ▼                              ▼
  ┌───────────────┐           ┌─────────────────────┐
  │  VectorStore  │           │  Node 1             │
  │ similarity    │           │  query_restructure  │
  │ search top-20 │           │                     │
  │ (semantic +   │           │  TopicExtractor     │
  │  keyword      │           │  (TF-IDF over       │
  │  hybrid)      │           │  corpus) → topics   │
  └───────┬───────┘           │                     │
          │                   │  LLM scope check:   │
          ▼                   │  in-scope → rewrite │
  ┌───────────────┐           │  off-topic →        │
  │  AnswerGen    │           │  NOT_ANSWERABLE      │
  │  DeepSeek V3  │           └──────────┬──────────┘
  │  via OpenRouter│                     │
  │               │           ┌──────────▼──────────┐
  │  Answer only  │           │  Node 2             │
  │  from context │           │  vector_search      │
  └───────┬───────┘           │                     │
          │                   │  VectorStore        │
          │                   │  similarity search  │
          │                   │  top-10             │
          │                   │  (semantic +        │
          │                   │   keyword hybrid)   │
          │                   └──────────┬──────────┘
          │                              │
          │                   ┌──────────▼──────────┐
          │                   │  Node 3             │
          │                   │  validate_relevance │
          │                   │                     │
          │                   │  LLM judges each    │
          │                   │  chunk: keep or     │
          │                   │  discard            │
          │                   │  Returns JSON:      │
          │                   │  {relevant_excerpts}│
          │                   └──────────┬──────────┘
          │                              │
          │                   ┌──────────▼──────────┐
          │                   │  Node 4             │
          │                   │  generate_answer    │
          │                   │                     │
          │                   │  LLM answers from   │
          │                   │  validated excerpts │
          │                   │  only               │
          │                   └──────────┬──────────┘
          │                              │
          └──────────────┬───────────────┘
                         ▼
          ┌──────────────────────────────┐
          │   CombinedAnswerResponse     │
          │                              │
          │  traditional_rag: {          │
          │    answer, confidence_score, │
          │    latency_seconds, cost_usd,│
          │    input_tokens,             │
          │    output_tokens, sources    │
          │  }                           │
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
| `embedder` | `embedder.py` | OpenAI `text-embedding-3-small/large` via direct API key or OpenRouter fallback |
| `TopicExtractor` | `topic_extractor.py` | Pure-Python TF-IDF over the full corpus. Extracts ~23 high-signal topic terms (bigrams + unigrams). Cache-aware: auto-refreshes when Qdrant point count changes after ingest |
| `config` | `config.py` | All env-var driven settings (models, keys, chunk sizes, pricing) |

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

### Approach 2 — Agentic RAG (`approach2_agents/`)

Four-node LangGraph graph. Each node is a synchronous function run via LangGraph's async executor.

```
START
  │
  ▼
[restructure_query]  ←── TopicExtractor.get_topics(store)
  │  Scope check: is this question in the corpus?
  │  YES → rewrite query for better VDB recall
  │  NO  → set restructured_query = "NOT_ANSWERABLE"
  │
  ▼
[retrieve_docs]
  │  If NOT_ANSWERABLE → skip, return empty
  │  Else → VectorStore.similarity_search(restructured_query, top_k=10)
  │
  ▼
[validate_relevance]
  │  If empty docs → skip
  │  Else → LLM judges each chunk, returns only relevant ones as JSON
  │
  ▼
[generate_answer]
  │  If NOT_ANSWERABLE → return "This question cannot be answered with the available data."
  │  Else → LLM answers from validated excerpts only
  │
  END
```

**State schema (`QAState`):**

| Field | Type | Description |
|---|---|---|
| `question` | `str` | Original user question, never mutated |
| `restructured_query` | `str` | Rewritten query or `NOT_ANSWERABLE` |
| `retrieved_docs` | `str` | Raw formatted excerpts from Qdrant |
| `validated_docs` | `str` | Filtered excerpts after relevance check |
| `sources` | `list[str]` | Unique source filenames |
| `answer` | `str` | Final answer |
| `confidence_score` | `float` | Avg cosine similarity of retrieved chunks |
| `input_tokens` | `int` | Accumulated across all 3 LLM nodes |
| `output_tokens` | `int` | Accumulated across all 3 LLM nodes |

### TF-IDF Topic Extractor (`shared/topic_extractor.py`)

Prevents the query-restructure LLM from generating hallucinated queries for off-topic questions.

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

**Scope check in the restructure prompt:**

The topics list is injected into the LLM prompt alongside a plain-language description of the corpus domains:
- International criminal law (ICC, ICTY, KSC)
- Criminal investigations: fraud, financial crimes, corruption
- Criminal procedure: testimony, disclosure, detention, reparations
- Norwegian / Nordic criminal law traditions
- Political accountability (Council of Europe, Azerbaijan)
- Philosophy and foundations of criminal law
- Professional integrity and criticism of justice institutions

The LLM is instructed to return `NOT_ANSWERABLE` **only** for questions completely outside the legal domain (cooking, weather, sports, etc.). For anything with plausible legal relevance it rewrites the query.

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
│   ├── app.py                          # FastAPI app, lifespan, all endpoints
│   ├── shared/
│   │   ├── config.py                   # All env-var settings
│   │   ├── embedder.py                 # OpenAI embeddings (direct or via OpenRouter)
│   │   ├── pdf_parser.py               # PDF → chunks (pymupdf4llm + text splitter)
│   │   ├── vector_store.py             # Qdrant wrapper (ingest, search, scroll)
│   │   └── topic_extractor.py          # TF-IDF corpus topic extraction + cache
│   ├── approach1_rag/
│   │   ├── pipeline.py                 # Orchestrates retrieve → generate
│   │   ├── retriever.py                # Hybrid vector + keyword search
│   │   ├── answer_generator.py         # DeepSeek V3 answer synthesis
│   │   └── confidence_scorer.py        # Avg cosine similarity scorer
│   └── approach2_agents/
│       ├── orchestrator.py             # LangGraph graph, 4-node pipeline
│       └── tools/
│           ├── query_restructure_tool.py   # Node 1: scope check + query rewrite
│           ├── vector_search_tool.py       # Node 2: Qdrant search
│           └── relevance_validator_tool.py # Node 3: LLM relevance filter
├── pyproject.toml
└── README.md
```

---

## Models Used

| Role | Model | Via |
|---|---|---|
| Answer generation (both approaches) | `deepseek/deepseek-v3.2` | OpenRouter |
| Query restructure (approach 2) | `deepseek/deepseek-v3.2` | OpenRouter |
| Relevance validation (approach 2) | `deepseek/deepseek-v3.2` | OpenRouter |
| Embeddings | `text-embedding-3-small` (default) | OpenAI direct or OpenRouter |

DeepSeek V3 context window: **64 K tokens**. The full restructure system prompt (corpus description + ~23 TF-IDF topics) uses ~530 tokens.

---

## Approach Comparison

| | Traditional RAG | Agentic RAG |
|---|---|---|
| LLM calls per query | 1 | 2–3 |
| Retrieval query | Raw user question | LLM-rewritten for VDB recall |
| Chunk filtering | None (top-20 passed to LLM) | LLM validates relevance |
| Off-topic handling | Answers "not found" after retrieval | Detects before retrieval, skips all LLM calls |
| Typical latency | ~1–4 s | ~5–10 s |
| Typical cost | Lower | Higher |
| Answer quality | Good on simple factual queries | Better on complex/ambiguous queries |
