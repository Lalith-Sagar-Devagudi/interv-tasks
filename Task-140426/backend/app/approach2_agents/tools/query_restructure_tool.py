from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Injected at startup by the orchestrator
_llm: ChatOpenAI | None = None

_SYSTEM = (
    "You are a legal research assistant. "
    "Rewrite the user's question into a precise search query for a legal document vector database. "
    "Rules:\n"
    "- Keep the output as a natural-language sentence or short paragraph, NOT a keyword list.\n"
    "- Expand legal terminology and include relevant synonyms where helpful.\n"
    "- Preserve the original question's intent — do not change what is being asked.\n"
    "- Return ONLY the rewritten query — no explanation, no quotes, no extra text."
)


def init_restructure_tool(llm: ChatOpenAI) -> None:
    """Inject the LLM instance before the tool is called."""
    global _llm
    _llm = llm


def _do_restructure(llm: ChatOpenAI, question: str) -> tuple[str, dict]:
    """Core logic — sync.  Returns (restructured_query, usage_metadata)."""
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"Original question: {question}"),
    ]
    print(messages)
    response = llm.invoke(messages)
    print(response)
    usage = response.usage_metadata or {}
    restructured = response.content.strip()
    print(f"[tool:query_restructure] {question!r} → {restructured!r}")
    return restructured, usage


@tool
def query_restructure(question: str) -> str:
    """Restructure and enrich a user question into a targeted vector DB query.

    Rewrites the question to maximise recall from a legal document corpus —
    expanding legal terminology, adding synonyms, and focusing on key entities.

    Args:
        question: The original user question.

    Returns:
        A restructured query string optimised for semantic similarity search.
    """
    if _llm is None:
        return question
    return _do_restructure(_llm, question)[0]
