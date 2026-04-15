"""TF-IDF based topic extractor for the legal document corpus.

Scrolls all chunks stored in the vector store, runs a pure-Python TF-IDF
algorithm (no external ML libraries required) and returns a compact list of
high-level concepts that describe what the indexed documents actually cover.

Design decisions informed by empirical runs on the actual 35-document corpus
(~1M tokens, mix of ICC/KSC rulings and TOAEP academic books):

  - Document-level TF-IDF  (one doc = all its chunks merged)
  - Per-doc top-K approach for unigrams so minority documents (small rulings)
    aren't drowned out by large academic books.
  - Bigram candidates are gated: both constituent words must independently
    appear in the top-N unigram pool, which eliminates boilerplate phrases
    like "publication series" or author names that pollute a global bigram run.
  - Strict stopword list covering three layers:
      1. Generic English function words
      2. Legal-document structural boilerplate (appears in every doc)
      3. Corpus-specific metadata noise (month names, TOAEP editor names, etc.)
  - Singular/plural deduplication via suffix heuristic.
  - Overlapping-bigram deduplication (keeps higher-scoring of sharing-word pairs).
  - Target: 22–25 concepts ≈ 80–100 prompt tokens — tiny relative to
    DeepSeek V3's 64 K context window, but enough signal for scope-checking.

The extractor is cache-aware: it tracks the Qdrant point count and refreshes
automatically after each /ingest without any external invalidation call.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.vector_store import VectorStore


# ─── Stopwords ───────────────────────────────────────────────────────────────
# Layer 1: generic English (function words / short connectors)
_SW_ENGLISH = {
    "about", "above", "after", "again", "against", "being", "below",
    "between", "both", "could", "doing", "during", "each", "further",
    "having", "here", "itself", "just", "more", "most", "myself",
    "other", "otherwise", "over", "same", "should", "since", "some",
    "such", "than", "that", "their", "them", "then", "there", "these",
    "they", "this", "those", "through", "under", "until", "very",
    "were", "what", "when", "where", "which", "while", "whom",
    "will", "with", "would", "your",
}

# Layer 2: legal-document structural boilerplate present in virtually every doc
_SW_LEGAL_STRUCTURE = {
    # Statutory language
    "shall", "upon", "pursuant", "whereby", "whereas", "herein", "hereof",
    "hereto", "hereby", "thereof", "thereto", "notwithstanding", "according",
    "provided", "including", "within",
    # Document structure terms
    "section", "article", "paragraph", "paragraphs", "chapter", "clause",
    "annex", "document", "page", "pages", "submission", "submissions",
    "reference",
    # Court actors that appear in every single case header (no topical signal)
    "judge", "judges", "counsel", "registry",
    # Court outputs (every doc has at least one of these — no topical value)
    "court", "judgment", "decision", "order", "tribunal",
    # Generic party labels
    "party", "parties",
    # Procedural verbs that appear everywhere
    "request", "requests", "extension", "extensions",
}

# Layer 3: corpus-specific metadata and form-template noise
# (discovered empirically from actual TF-IDF runs on this corpus)
_SW_CORPUS_NOISE = {
    # TOAEP academic book metadata (author/editor names + publishing words)
    "publication", "series", "editors", "edited", "author", "authors",
    "publisher", "toaep", "isbn", "volume", "edition", "printed",
    "english", "classification", "quality", "control",
    # Editor/author name fragments that pollute bigrams
    "carsten", "stahn", "fidelma", "donlon", "antonio", "angotti",
    "fahsing", "tashukov", "penneman", "willems", "schreurs",
    "roumen", "nenkov", "piotr", "vidar", "stensland",
    # Form-template artifacts (evidence submission forms)
    "redacted", "certify", "witnessing", "language", "registration",
    "number", "given", "false", "place", "picture", "video", "email",
    "version", "corrected", "contents", "subject", "object",
    # Generic procedural words with no topical specificity
    "confirmed", "officer", "officers", "specialist", "associated",
    "materials", "prior", "recorded", "noise",
    # Redundant plural / abbreviation of other kept terms
    "paras", "testimonies",
    # Too generic as standalone words (subsumed by bigrams or context)
    "country", "norway",    # keep "norwegian" — more specific; norway is redundant
    "declaration", "limit", # procedural artifacts
    "understands",          # from KSC witness-form template ("understands the language")
    "himself", "herself", "themselves",  # generic pronouns that leak through on long texts
    # Calendar metadata
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
    # Recurring location boilerplate in ICC/KSC headers
    "hague", "netherlands", "bangui",
    # Recurring abbreviation from ICC Legal Tools Database cross-links
    "tools",
}

_STOPWORDS = _SW_ENGLISH | _SW_LEGAL_STRUCTURE | _SW_CORPUS_NOISE

# ─── Tunable constants ────────────────────────────────────────────────────────
_MIN_LEN      = 5   # ignore tokens shorter than this
_MIN_DF       = 4   # term must appear in at least 4 docs (≈ 11 % of 35-doc corpus)
_TOP_PER_DOC  = 8   # per-document top-N terms fed into the coverage counter
                    # (raised from 6 so conceptual terms like integrity/criticism
                    #  in essay-heavy docs compete with procedural terms)
_UNI_POOL_K   = 100 # unigram candidate pool size (gating bigram construction)
_MIN_BG_DF    = 5   # bigram must appear in at least 5 docs (≈ 14 %)
_MAX_BIGRAMS  = 8   # bigrams in final output
_MAX_UNIGRAMS = 17  # unigrams in final output (after subtracting bigram words)


# ─── Core NLP helpers ────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    raw = re.findall(r"\b[a-zA-Z]{" + str(_MIN_LEN) + r",}\b", text.lower())
    return [t for t in raw if t not in _STOPWORDS]


def _stem_key(term: str) -> str:
    """Lightweight suffix normaliser for deduplication.

    Strips common English plural/gerund suffixes so 'investigation' and
    'investigations' map to the same key, as do 'norway'/'norwegian'.
    Not a full stemmer — just enough to collapse obvious redundant pairs.
    """
    for suffix in ("tions", "tion", "ings", "ing", "ies", "ian", "ans", "ans", "es", "s"):
        if term.endswith(suffix) and len(term) - len(suffix) >= _MIN_LEN:
            return term[: -len(suffix)]
    return term


# ─── TF-IDF computation ───────────────────────────────────────────────────────

def _compute_global_df(tokenized_docs: list[list[str]]) -> Counter:
    df: Counter = Counter()
    for doc in tokenized_docs:
        for t in set(doc):
            df[t] += 1
    return df


def _per_doc_unigram_coverage(
    tokenized_docs: list[list[str]],
    idf: dict[str, float],
) -> Counter:
    """Count how many documents rank each term in their top-K by TF-IDF.

    This gives all documents equal weight regardless of length, so a short
    three-page ruling contributes the same number of terms as a 300-page book.
    """
    coverage: Counter = Counter()
    for doc in tokenized_docs:
        tf = Counter(doc)
        L  = max(len(doc), 1)
        scores = {t: (c / L) * idf[t] for t, c in tf.items() if t in idf}
        for t in sorted(scores, key=scores.get, reverse=True)[: _TOP_PER_DOC]:
            coverage[t] += 1
    return coverage


def _bigram_scores(
    tokenized_docs: list[list[str]],
    uni_candidates: set[str],
    idf: dict[str, float],
    N: int,
) -> Counter:
    """TF-IDF scores for bigrams whose both words are in uni_candidates.

    Document frequency is counted on distinct-per-doc occurrences to avoid
    inflating scores from repetitive template sections.
    """
    bg_df: Counter = Counter()
    for doc in tokenized_docs:
        seen: set[str] = set()
        for i in range(len(doc) - 1):
            w1, w2 = doc[i], doc[i + 1]
            if w1 in uni_candidates and w2 in uni_candidates:
                bg = f"{w1} {w2}"
                if bg not in seen:
                    bg_df[bg] += 1
                    seen.add(bg)

    bg_scores: Counter = Counter()
    for doc in tokenized_docs:
        bgs = [
            f"{doc[i]} {doc[i+1]}"
            for i in range(len(doc) - 1)
            if doc[i] in uni_candidates and doc[i + 1] in uni_candidates
        ]
        tf = Counter(bgs)
        L  = max(len(bgs), 1)
        for bg, c in tf.items():
            if bg_df[bg] < _MIN_BG_DF:
                continue
            w1, w2 = bg.split()
            bi_idf = (idf.get(w1, 0.0) + idf.get(w2, 0.0)) / 2.0
            bg_scores[bg] += (c / L) * bi_idf

    return bg_scores


def _dedup_overlapping_bigrams(bigrams: list[str]) -> list[str]:
    """Remove lower-scoring bigrams that share a word with a higher-scoring one.

    E.g. if 'criminal investigation' and 'investigation procedure' both appear,
    only 'criminal investigation' (ranked higher) is kept.
    """
    kept: list[str] = []
    dropped: set[str] = set()
    for i, a in enumerate(bigrams):
        if a in dropped:
            continue
        kept.append(a)
        wa = a.split()
        for b in bigrams[i + 1:]:
            wb = b.split()
            if wa[1] == wb[0] or wb[1] == wa[0]:
                dropped.add(b)
    return kept


def _dedup_stems(terms: list[str]) -> list[str]:
    """Keep only the first occurrence of each stem-normalised form."""
    out: list[str] = []
    seen_stems: set[str] = set()
    for t in terms:
        stem = _stem_key(t)
        if stem not in seen_stems:
            out.append(t)
            seen_stems.add(stem)
            seen_stems.add(t)
    return out


# ─── Public extraction function ───────────────────────────────────────────────

def extract_topics_from_texts(texts: list[str]) -> list[str]:
    """Given a list of raw text strings (one per document / chunk), return a
    compact list of high-level domain concepts ranked by corpus relevance.

    Returns at most ``_MAX_BIGRAMS + _MAX_UNIGRAMS`` items.
    """
    if not texts:
        return []

    N          = len(texts)
    tokenized  = [_tokenize(t) for t in texts]
    df         = _compute_global_df(tokenized)

    # IDF: terms must appear in [_MIN_DF, ∞) docs — no hard upper-DF cap;
    # the log-dampening in IDF naturally down-weights corpus-ubiquitous terms.
    idf = {
        t: math.log((N + 1) / (c + 1))
        for t, c in df.items()
        if c >= _MIN_DF
    }

    # ── Step 1: per-doc coverage → unigram candidate pool ─────────────────
    coverage       = _per_doc_unigram_coverage(tokenized, idf)
    uni_candidates = {t for t, _ in coverage.most_common(_UNI_POOL_K)}

    # ── Step 2: bigrams gated by uni_candidates ────────────────────────────
    bg_scores = _bigram_scores(tokenized, uni_candidates, idf, N)
    top_bgs_raw   = [b for b, _ in bg_scores.most_common(_MAX_BIGRAMS * 3)]
    top_bgs_dedup = _dedup_overlapping_bigrams(top_bgs_raw)
    final_bgs     = _dedup_stems(top_bgs_dedup)[: _MAX_BIGRAMS]

    # ── Step 3: unigrams — fill remaining slots with non-redundant coverage ─
    used_words = {w for bg in final_bgs for w in bg.split()}
    uni_pool   = [t for t, _ in coverage.most_common(60) if t not in used_words]
    final_unis = _dedup_stems(uni_pool)[: _MAX_UNIGRAMS]

    return final_bgs + final_unis


# ─── Cache-aware wrapper ──────────────────────────────────────────────────────

class TopicExtractor:
    """Lazily extracts and caches corpus topics from the live Qdrant store.

    Re-extracts automatically whenever the chunk count changes (i.e. after a
    new /ingest call).  No manual cache invalidation needed.

    Context window budget (DeepSeek V3 — 64 K tokens):
      ~23 topic terms × avg 3 tokens ≈ 90 tokens for the topic list.
      The full restructure system prompt stays under ~300 tokens.
    """

    def __init__(self) -> None:
        self._cached_topics: list[str] = []
        self._cached_point_count: int  = -1

    def get_topics(self, vector_store: "VectorStore") -> list[str]:
        """Return cached topics, refreshing if the corpus has changed."""
        current_count = vector_store.get_point_count()
        if current_count != self._cached_point_count:
            print(
                f"[topic_extractor] Corpus changed "
                f"({self._cached_point_count} → {current_count} chunks). "
                "Re-extracting topics …"
            )
            # Use document-level TF-IDF: merge all chunks from the same source
            # PDF back into a single text before running TF-IDF.  This ensures
            # equal document weight regardless of length (a 3-page ruling and a
            # 300-page book each count as one document), preventing long books
            # like Leviathan from dominating the topic scores.
            doc_texts = vector_store.get_texts_grouped_by_source()
            self._cached_topics = extract_topics_from_texts(doc_texts)
            self._cached_point_count = current_count
            print(
                f"[topic_extractor] Extracted {len(self._cached_topics)} topics: "
                + ", ".join(self._cached_topics[:8]) + " …"
            )
        return self._cached_topics

    def invalidate(self) -> None:
        """Force re-extraction on the next call to get_topics()."""
        self._cached_point_count = -1
