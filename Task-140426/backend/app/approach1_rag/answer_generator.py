from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from shared.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL, LLM_MAX_TOKENS

_SYSTEM = """You are a precise legal document analyst.
Answer the user's question based strictly on the provided document excerpts.

Rules:
- Use only information present in the context — do not hallucinate.
- If the answer cannot be found, respond: "The provided documents do not contain sufficient information to answer this question."
- Be concise; cite clause numbers or section headings when available."""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
])


class AnswerGenerator:
    """Calls DeepSeek V3 via OpenRouter to produce the final answer."""

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            max_tokens=LLM_MAX_TOKENS,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        self._chain = _PROMPT | self._llm

    def generate(self, question: str, context: str) -> tuple[str, dict]:
        """Returns (answer_text, usage_metadata).  Sync — call via asyncio.to_thread.

        usage_metadata keys: input_tokens, output_tokens, total_tokens.
        """
        print(f"[answer_generator] Sending request to LLM model={LLM_MODEL!r} ...")
        print("[answer_generator] ── FULL PROMPT ──────────────────────────────")
        print(f"[answer_generator] SYSTEM:\n{_SYSTEM}")
        print(f"[answer_generator] QUESTION:\n{question}")
        print(f"[answer_generator] CONTEXT ({len(context)} chars):\n{context}")
        print("[answer_generator] ── END PROMPT ──────────────────────────────")
        response = self._chain.invoke({"question": question, "context": context})
        usage = response.usage_metadata or {}
        print(f"[answer_generator] LLM responded ({len(response.content)} chars)  usage={usage}")
        return response.content, usage
