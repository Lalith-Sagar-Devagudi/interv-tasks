"""Legal QA API — entry point.

Endpoints
─────────
  POST /ingest          Upload PDF files → parse → embed → store in Qdrant
  POST /ask?mode=rag    Traditional RAG  (Gemma 4 31B via OpenRouter)
  POST /ask?mode=agent  Agentic RAG      (DeepSeek V3 via OpenRouter + LangGraph)
  GET  /health          Liveness probe
"""

import asyncio
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Dict, List

from dotenv import load_dotenv

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

load_dotenv()

# Ensure local packages are importable when running with `uvicorn app:app`
import sys
sys.path.append(str(Path(__file__).parent))

from approach1_rag.pipeline import RAGPipeline
from approach2_agents.orchestrator import LegalQAOrchestrator
from auth import create_access_token, require_auth, verify_credentials
from shared.pdf_parser import parse_pdfs
from shared.vector_store import VectorStore


# ── App lifecycle ─────────────────────────────────────────────────────────────

vector_store: VectorStore | None = None
rag_pipeline: RAGPipeline | None = None
agent_orchestrator: LegalQAOrchestrator | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global vector_store, rag_pipeline, agent_orchestrator
    vector_store = VectorStore()
    rag_pipeline = RAGPipeline(vector_store)
    agent_orchestrator = LegalQAOrchestrator(vector_store)
    print("[startup] All components initialised.")
    yield
    print("[shutdown] Cleaning up.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Legal QA API",
    description=(
        "RAG-powered question answering over legal PDF documents.\n\n"
        "**Approach 1** — Traditional RAG: deterministic single-pass pipeline — "
        "retrieve docs from VDB → (query + docs) + LLM → answer. "
        "Uses **DeepSeek V3** via OpenRouter.\n\n"
        "**Approach 2** — Agentic RAG: four-node LangGraph pipeline — "
        "(1) **query_restructure**: scope-checks the question against TF-IDF corpus topics, "
        "rewrites it for better VDB recall, or short-circuits with NOT_ANSWERABLE; "
        "(2) **vector_search**: retrieves top-10 chunks using the enriched query; "
        "(3) **validate_relevance**: LLM filters chunks to only those that help answer the question; "
        "(4) **generate_answer**: final LLM call produces the answer. "
        "Uses **DeepSeek V3** via OpenRouter."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _fix_binary_schemas(obj: Any) -> None:
    """Replace OpenAPI 3.1 contentMediaType with 3.0-compatible format:binary."""
    if isinstance(obj, dict):
        if obj.get("type") == "string" and "contentMediaType" in obj:
            del obj["contentMediaType"]
            obj["format"] = "binary"
        for v in obj.values():
            _fix_binary_schemas(v)
    elif isinstance(obj, list):
        for item in obj:
            _fix_binary_schemas(item)


def custom_openapi() -> Dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema["openapi"] = "3.0.3"
    _fix_binary_schemas(schema)
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi  # type: ignore[method-assign]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, example="What is the effective date of the agreement between parties X and Y?")


class ApproachResult(BaseModel):
    answer: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    latency_seconds: float
    cost_usd: float
    input_tokens: int
    output_tokens: int
    sources: list[str] = Field(default_factory=list)


class CombinedAnswerResponse(BaseModel):
    question: str
    traditional_rag: ApproachResult
    agentic_rag: ApproachResult


# ── /auth ────────────────────────────────────────────────────────────────────

@app.post(
    "/auth/token",
    response_model=TokenResponse,
    tags=["auth"],
    summary="Log in and get a Bearer token",
)
async def login(form: Annotated[OAuth2PasswordRequestForm, Depends()]):
    """Exchange username + password for a signed JWT Bearer token.

    Use the returned token in the `Authorization: Bearer <token>` header
    (or click the **Authorize** button in Swagger UI) to access all other endpoints.

    Credentials are configured via environment variables:
    - `AUTH_USERNAME` (default: `admin`)
    - `AUTH_PASSWORD` (required)
    - `JWT_SECRET_KEY` (required — long random string)
    - `JWT_EXPIRE_MINUTES` (default: `60`)
    """
    if not verify_credentials(form.username, form.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(form.username)
    return TokenResponse(access_token=token)


# ── /ingest ───────────────────────────────────────────────────────────────────

@app.post(
    "/ingest",
    summary="Upload and index legal PDF documents",
    response_description="Number of text chunks indexed into the vector store",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["files"],
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                                "maxItems": 10,
                                "description": "Up to 10 PDF files",
                            }
                        },
                    }
                }
            },
        }
    },
)
async def ingest(
    files: List[UploadFile] = File(...),
    _: Annotated[str, Depends(require_auth)] = "",
):
    """Upload one or more legal PDF files.

    The API will:
    1. Extract text from each PDF using PyMuPDF.
    2. Split the text into overlapping chunks (1 500 tokens, 200 overlap).
    3. Embed each chunk with OpenAI `text-embedding-3-small`.
    4. Upsert all vectors into the Qdrant `legal_docs` collection.

    Subsequent calls **add** documents to the existing collection — they do
    not replace it, allowing incremental ingestion.
    """
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not ready.")

    MAX_FILES = 10
    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files: received {len(files)}, maximum allowed is {MAX_FILES}.",
        )

    non_pdf = [f.filename for f in files if not (f.filename or "").endswith(".pdf")]
    if non_pdf:
        raise HTTPException(status_code=400, detail=f"Only PDF files are accepted: {non_pdf}")

    print(f"\n[ingest] ── Starting ingestion of {len(files)} file(s) ──")
    for i, f in enumerate(files, 1):
        print(f"[ingest]   {i}. {f.filename}")

    tmp_dir = tempfile.mkdtemp()
    try:
        print(f"[ingest] Step 1/3 — Saving uploads to temp dir: {tmp_dir}")
        pdf_paths: list[str] = []
        for upload in files:
            dest = Path(tmp_dir) / (upload.filename or "upload.pdf")
            with dest.open("wb") as fh:
                shutil.copyfileobj(upload.file, fh)
            size_kb = dest.stat().st_size / 1024
            print(f"[ingest]   Saved {upload.filename} ({size_kb:.1f} KB)")
            pdf_paths.append(str(dest))

        print(f"[ingest] Step 2/3 — Parsing PDFs (1 chunk per page) ...")
        documents = parse_pdfs(pdf_paths)
        if not documents:
            raise HTTPException(
                status_code=422,
                detail="No extractable text found in the uploaded PDFs.",
            )
        print(f"[ingest]   Total pages extracted: {len(documents)}")

        print(f"[ingest] Step 3/3 — Embedding & upserting {len(documents)} chunks into Qdrant ...")
        count = vector_store.add_documents(documents)
        print(f"[ingest] ── Done. {count} chunks indexed. ──\n")
        return {
            "message": "Indexing complete.",
            "files_processed": [f.filename for f in files],
            "chunks_indexed": count,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── /ask ──────────────────────────────────────────────────────────────────────

@app.post(
    "/ask",
    response_model=CombinedAnswerResponse,
    summary="Ask a question — runs both RAG approaches in parallel",
)
async def ask(
    request: QuestionRequest,
    _: Annotated[str, Depends(require_auth)] = "",
):
    """Ask a natural-language question over the indexed legal documents.

    Both approaches run **in parallel** and their results are returned together
    so you can compare answers, confidence, latency, and cost side-by-side.

    Example input:
    ```json
    { "question": "What is the effective date of the agreement between parties X and Y?" }
    ```

    Example output:
    ```json
    {
      "question": "...",
      "traditional_rag": {
        "answer": "...", "confidence_score": 0.82,
        "latency_seconds": 1.4, "cost_usd": 0.00021,
        "input_tokens": 750, "output_tokens": 120, "sources": [...]
      },
      "agentic_rag": {
        "answer": "...", "confidence_score": 0.79,
        "latency_seconds": 6.2, "cost_usd": 0.00089,
        "input_tokens": 2800, "output_tokens": 410, "sources": [...]
      }
    }
    ```
    """
    print(f"\n{'='*60}")
    print(f"[ask] ▶ question={request.question!r} — running both approaches in parallel")
    print(f"{'='*60}")

    if rag_pipeline is None or agent_orchestrator is None:
        raise HTTPException(status_code=503, detail="Pipelines not ready.")

    try:
        rag_result, agent_result = await asyncio.gather(
            rag_pipeline.run(request.question),
            agent_orchestrator.run(request.question),
        )

        print(f"[ask] ◀ traditional_rag latency={rag_result['latency_seconds']}s  cost=${rag_result['cost_usd']:.8f}")
        print(f"[ask] ◀ agentic_rag     latency={agent_result['latency_seconds']}s  cost=${agent_result['cost_usd']:.8f}")
        print(f"{'='*60}\n")

        return CombinedAnswerResponse(
            question=request.question,
            traditional_rag=ApproachResult(
                answer=rag_result["answer"],
                confidence_score=rag_result["confidence_score"],
                latency_seconds=rag_result["latency_seconds"],
                cost_usd=rag_result["cost_usd"],
                input_tokens=rag_result["input_tokens"],
                output_tokens=rag_result["output_tokens"],
                sources=rag_result.get("sources", []),
            ),
            agentic_rag=ApproachResult(
                answer=agent_result["answer"],
                confidence_score=agent_result["confidence_score"],
                latency_seconds=agent_result["latency_seconds"],
                cost_usd=agent_result["cost_usd"],
                input_tokens=agent_result["input_tokens"],
                output_tokens=agent_result["output_tokens"],
                sources=agent_result.get("sources", []),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── /debug ───────────────────────────────────────────────────────────────────

@app.post("/collection/reset", summary="Drop and recreate the vector collection (use before re-ingesting with a new embedding model)")
async def collection_reset(_: Annotated[str, Depends(require_auth)] = ""):
    """Deletes the existing Qdrant collection and recreates it empty with the
    current VECTOR_SIZE.  Run this once after changing the embedding model,
    then re-upload all documents via /ingest."""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not ready.")
    await asyncio.to_thread(vector_store.reset_collection)
    return {"message": f"Collection reset. Ready to ingest with VECTOR_SIZE={vector_store.embeddings.model}."}


@app.get("/debug/collection-info", summary="Show Qdrant collection vector dimensions (debug only)")
async def debug_collection_info(_: Annotated[str, Depends(require_auth)] = ""):
    """Return the vector config of the Qdrant collection so you can verify
    the stored dimension matches the current embedding model's output size."""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not ready.")
    collections = vector_store.client.get_collections().collections
    result = {}
    for col in collections:
        info = vector_store.client.get_collection(col.name)
        vectors_cfg = info.config.params.vectors
        result[col.name] = {
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "vector_size": vectors_cfg.size if hasattr(vectors_cfg, "size") else str(vectors_cfg),
            "distance": str(vectors_cfg.distance) if hasattr(vectors_cfg, "distance") else "unknown",
        }
    return result


@app.get("/debug/keyword", summary="Scan all chunks for a keyword (debug only)")
async def debug_keyword(q: str, limit: int = 5, _: Annotated[str, Depends(require_auth)] = ""):
    """Scroll the entire Qdrant collection for chunks whose text contains 'q'.
    Returns the raw chunk content so you can verify what was indexed."""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not ready.")
    matches = await asyncio.to_thread(vector_store.keyword_search, q, limit)
    return {"keyword": q, "matches_found": len(matches), "results": matches}


@app.get("/debug/semantic", summary="Run a high-k semantic search (debug only)")
async def debug_semantic(q: str, top_k: int = 50, _: Annotated[str, Depends(require_auth)] = ""):
    """Return the top-N semantically similar chunks for a query with their scores."""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not ready.")
    results = await asyncio.to_thread(
        vector_store.similarity_search_with_scores, q, top_k
    )
    return {
        "query": q,
        "results": [
            {
                "rank": i + 1,
                "source": doc.metadata.get("source"),
                "score": round(score, 4),
                "preview": doc.page_content[:200],
            }
            for i, (doc, score) in enumerate(results)
        ],
    }


# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health", summary="Liveness probe")
async def health():
    return {
        "status": "ok",
        "collection_ready": vector_store.collection_exists() if vector_store else False,
    }


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
