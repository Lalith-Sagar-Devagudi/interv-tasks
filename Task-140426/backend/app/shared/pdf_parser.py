from pathlib import Path

import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from shared.config import CHUNK_OVERLAP, CHUNK_SIZE


def _get_legal_chunker() -> RecursiveCharacterTextSplitter:
    """Legal-aware recursive chunker.

    Separator hierarchy respects legal document structure: tries to split on
    article/section/clause boundaries first, only falls back to
    paragraph/sentence if needed.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            # Legal structural boundaries — try these first
            "\n\nARTICLE",
            "\n\nArticle",
            "\n\nSECTION",
            "\n\nSection",
            "\n\nCLAUSE",
            "\n\nClause",
            "\n\n§",        # section symbol common in German/EU legal docs
            r"\n\n\d+\.",   # numbered clauses like "1. ", "12. "
            # Generic fallbacks
            "\n\n",
            "\n",
            ". ",
            " ",
        ],
        is_separator_regex=True,
    )


def parse_pdfs(file_paths: list[str]) -> list[Document]:
    """Extract text from PDFs and split into legal-aware chunks.

    Uses pymupdf4llm for high-fidelity markdown extraction, then applies a
    RecursiveCharacterTextSplitter with legal-document separator hierarchy
    (article → section → clause → paragraph → sentence) so chunk boundaries
    respect legal structure rather than arbitrary character counts.

    Args:
        file_paths: Absolute paths to PDF files.

    Returns:
        List of LangChain Documents with ``source``, ``chunk_index``, and
        ``total_chunks`` in metadata.
    """
    chunker = _get_legal_chunker()
    all_docs: list[Document] = []

    for path in file_paths:
        if not path.endswith(".pdf"):
            continue
        name = Path(path).name
        try:
            print(f"[parser]   Extracting text from '{name}' ...")
            page_chunks = pymupdf4llm.to_markdown(path, page_chunks=True)
            print(f"[parser]   '{name}' — {len(page_chunks)} pages found")

            # Join all pages into one document so the chunker can split across
            # page boundaries (legal clauses often span pages).
            full_text = "\n\n".join(
                chunk["text"] for chunk in page_chunks if chunk.get("text", "").strip()
            )

            raw_chunks = chunker.split_text(full_text)
            docs = [
                Document(
                    page_content=chunk,
                    metadata={
                        "source": name,
                        "chunk_index": i,
                        "total_chunks": len(raw_chunks),
                    },
                )
                for i, chunk in enumerate(raw_chunks)
                if len(chunk.strip()) > 50  # drop near-empty chunks
            ]
            all_docs.extend(docs)
            print(f"[parser]   '{name}' — {len(docs)} chunks ready")
        except Exception as exc:
            print(f"[parser]   ERROR processing '{name}': {exc}")

    return all_docs
