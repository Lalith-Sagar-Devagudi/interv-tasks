import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Injected at startup by the orchestrator
_llm: ChatOpenAI | None = None

_SYSTEM = """\
You are a senior legal analyst. You will be given a user question and a list of document excerpts.

Your task:
1. Read each excerpt carefully.
2. Identify which excerpts DIRECTLY help answer the question (relevant facts, clauses, dates, parties, obligations, etc.).
3. Discard excerpts that are off-topic or provide no useful information.

Respond with ONLY valid JSON — no markdown fences, no extra text:
{
  "relevant_excerpts": ["<full text of excerpt 1>", "<full text of excerpt 2>", ...],
  "reasoning": "<one sentence explaining what was kept and why>"
}

If no excerpts are relevant, return:
{"relevant_excerpts": [], "reasoning": "<why nothing was relevant>"}"""


def init_validator_tool(llm: ChatOpenAI) -> None:
    """Inject the LLM instance before the tool is called."""
    global _llm
    _llm = llm


def _do_validate(llm: ChatOpenAI, question: str, retrieved_docs: str) -> tuple[str, dict]:
    """Core logic — sync.  Returns (validated_docs_string, usage_metadata)."""
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"Question: {question}\n\nDocument excerpts:\n{retrieved_docs}"),
    ]

    response = llm.invoke(messages)
    usage = response.usage_metadata or {}
    raw = response.content.strip()
    print(f"[tool:validate_relevance] LLM raw ({len(raw)} chars): {raw[:200].replace(chr(10), ' ')!r}{'...' if len(raw) > 200 else ''}")

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
        excerpts: list[str] = parsed.get("relevant_excerpts", [])
        reasoning: str = parsed.get("reasoning", "")
        print(f"[tool:validate_relevance] Kept {len(excerpts)} relevant excerpts. Reasoning: {reasoning!r}")

        if not excerpts:
            return "", usage

        return "\n\n---\n\n".join(excerpts), usage

    except (json.JSONDecodeError, ValueError):
        print("[tool:validate_relevance] JSON parse failed — passing all docs through")
        return retrieved_docs, usage
