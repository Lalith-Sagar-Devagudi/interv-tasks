"""QueryRestructureAgent — sub-agent responsible for scope checking and query rewriting.

Receives the user's raw question, checks whether it falls within the legal corpus,
and rewrites it into a precise natural-language search query for the vector DB.
Returns the NOT_ANSWERABLE sentinel when the question is completely outside scope.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Sentinel propagated through the multi-agent pipeline when a question is out of scope.
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


class QueryRestructureAgent:
    """Specialized sub-agent for scope checking and query rewriting.

    Has its own LLM instance and system prompt. Operates independently of the
    other sub-agents — the orchestrator calls it as a black box.
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm

    def run(self, question: str, topics: list[str] | None = None) -> tuple[str, dict]:
        """Scope-check and rewrite the question.

        Returns:
            (restructured_query, usage_metadata)
            restructured_query is either a rewritten search string or NOT_ANSWERABLE.
        """
        system_prompt = _build_system_prompt(topics)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Original question: {question}"),
        ]
        response = self._llm.invoke(messages)
        usage = response.usage_metadata or {}
        restructured = response.content.strip()
        print(f"[QueryRestructureAgent] {question!r} → {restructured!r}  usage={usage}")
        return restructured, usage
