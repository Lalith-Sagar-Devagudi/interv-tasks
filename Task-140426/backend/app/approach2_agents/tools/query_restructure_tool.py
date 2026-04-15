from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Injected at startup by the orchestrator
_llm: ChatOpenAI | None = None

# Sentinel value returned (and propagated through graph state) when the
# question cannot be answered from the available corpus.
NOT_ANSWERABLE = "NOT_ANSWERABLE"

_CORPUS_DESCRIPTION = """\
The corpus is a collection of legal documents and academic books covering:
- International criminal law: ICC, ICTY, Kosovo Specialist Chambers (KSC) rulings \
and proceedings
- Criminal investigations: fraud, financial crimes, corruption, investigative \
methodology and hypotheses
- Criminal procedure: witness testimony, evidence disclosure, pre-trial detention, \
defence rights, victims and reparations
- Constitutional proceedings: KSC constitutional panel referrals
- Norwegian and Nordic criminal law traditions, integrity in justice institutions
- Political accountability: Council of Europe, Azerbaijan, selective engagement
- Children's rights and crime prevention in Nordic countries
- Philosophy and foundations of criminal law (punishment theory, Bentham, Locke, Hobbes)
- Professional integrity and criticism of international justice institutions\
"""

_SYSTEM_BASE = """\
You are a legal research assistant with access to a specific document corpus.

{corpus_description}

The following key terms were extracted from the corpus by TF-IDF to give you
finer-grained signal about what is covered:
{{topics}}

Your task has two steps:

Step 1 — Scope check (be GENEROUS):
Respond with exactly NOT_ANSWERABLE ONLY if the question is completely outside the
legal domain described above — for example: cooking, weather, sports, fiction,
unrelated technology, or general science.
If the question has any plausible connection to law, criminal justice, human rights,
institutional integrity, or the specific topics listed, treat it as answerable.
The topic list is indicative, not exhaustive — match thematically, not literally.

Step 2 — Query rewrite (only if answerable):
Rewrite the question into a precise natural-language search query for a legal \
document vector database.
Rules:
- Output a natural-language sentence or short paragraph — NOT a keyword list.
- Expand legal terminology and include relevant synonyms where helpful.
- Preserve the original intent — do not change what is being asked.
- Return ONLY the rewritten query or NOT_ANSWERABLE — no explanation, no quotes, \
no extra text.
""".format(corpus_description=_CORPUS_DESCRIPTION)


def _build_system_prompt(topics: list[str] | None) -> str:
    """Combine the base system prompt with the current corpus topic list."""
    if not topics:
        # No topic context yet — fall back to plain rewrite with broad corpus hint
        return (
            "You are a legal research assistant with access to a corpus of legal "
            "documents and academic books on international criminal law, criminal "
            "investigations, and justice institutions. "
            "Rewrite the user's question into a precise search query for a legal "
            "document vector database.\n"
            "Rules:\n"
            "- Keep the output as a natural-language sentence or short paragraph, "
            "NOT a keyword list.\n"
            "- Expand legal terminology and include relevant synonyms where helpful.\n"
            "- Preserve the original question's intent.\n"
            "- Return ONLY the rewritten query — no explanation, no quotes, no extra text."
        )
    topic_str = ", ".join(topics)
    return _SYSTEM_BASE.replace("{topics}", topic_str)


def init_restructure_tool(llm: ChatOpenAI) -> None:
    """Inject the LLM instance before the tool is called."""
    global _llm
    _llm = llm


def _do_restructure(
    llm: ChatOpenAI,
    question: str,
    topics: list[str] | None = None,
) -> tuple[str, dict]:
    """Core logic — sync.

    Returns:
        (restructured_query, usage_metadata)

        restructured_query is either a rewritten search string or the
        ``NOT_ANSWERABLE`` sentinel when the question is out of corpus scope.
    """
    system_prompt = _build_system_prompt(topics)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Original question: {question}"),
    ]
    response = llm.invoke(messages)
    usage = response.usage_metadata or {}
    restructured = response.content.strip()
    print(f"[tool:query_restructure] {question!r} → {restructured!r}")
    return restructured, usage
