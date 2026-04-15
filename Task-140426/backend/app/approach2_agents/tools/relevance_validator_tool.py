"""RelevanceValidatorAgent — sub-agent that filters retrieved chunks by relevance.

Receives the original question and a set of retrieved document excerpts,
then asks an LLM to keep only the excerpts that directly help answer the question.
Returns only the relevant excerpts as a formatted string.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

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


class RelevanceValidatorAgent:
    """Specialized sub-agent for filtering retrieved chunks by relevance.

    Has its own LLM instance and system prompt. Operates independently of the
    other sub-agents — the orchestrator calls it as a black box.
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm

    def run(self, question: str, retrieved_docs: str) -> tuple[str, dict]:
        """Filter retrieved excerpts to only those relevant to the question.

        Returns:
            (validated_docs_string, usage_metadata)
            validated_docs_string is a newline-separated string of relevant excerpts,
            or an empty string if nothing was relevant.
        """
        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=f"Question: {question}\n\nDocument excerpts:\n{retrieved_docs}"),
        ]
        response = self._llm.invoke(messages)
        usage = response.usage_metadata or {}
        raw = response.content.strip()
        print(
            f"[RelevanceValidatorAgent] LLM raw ({len(raw)} chars): "
            f"{raw[:200].replace(chr(10), ' ')!r}{'...' if len(raw) > 200 else ''}"
        )

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
            print(
                f"[RelevanceValidatorAgent] Kept {len(excerpts)} relevant excerpts. "
                f"Reasoning: {reasoning!r}"
            )
            return ("\n\n---\n\n".join(excerpts) if excerpts else ""), usage

        except (json.JSONDecodeError, ValueError):
            print("[RelevanceValidatorAgent] JSON parse failed — passing all docs through")
            return retrieved_docs, usage
