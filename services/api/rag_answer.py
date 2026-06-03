import os
import re
import psycopg
import requests
from sentence_transformers import SentenceTransformer

# -----------------------------
# Config
# -----------------------------
DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/litreview",
)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims
embedder = SentenceTransformer(EMBED_MODEL_NAME)

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

CITE_RE = re.compile(r"\[C(\d+)\s+p(\d+)\]")


# -----------------------------
# Helpers
# -----------------------------
def embed(text: str) -> list[float]:
    """Text -> embedding vector (list[float])"""
    vec = embedder.encode(text, normalize_embeddings=True)
    return vec.tolist()


def is_junk(text: str) -> bool:
    """Heuristic filter for figure/number-heavy chunks that confuse the LLM."""
    t = " ".join((text or "").split())
    if len(t) < 120:
        return True
    letters = sum(ch.isalpha() for ch in t)
    ratio = letters / max(1, len(t))
    return ratio < 0.45


def retrieve(paper_id: int, query: str, top_k: int = 15):
    """Return list of (chunk_id, page, content, cosine_distance)"""
    qvec = embed(query)
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, page_start, content,
                       (embedding <=> %s::vector) AS cosine_distance
                FROM chunks
                WHERE paper_id = %s
                ORDER BY cosine_distance ASC
                LIMIT %s
                """,
                (qvec, paper_id, top_k),
            )
            return cur.fetchall()


def build_prompt(query: str, hits) -> str:
    """
    Ask for one-claim-per-line with a final citation.
    This is much more reliable than quote-exact templates on local LLMs + PDFs.
    """
    sources = []
    for chunk_id, page, content, dist in hits:
        snippet = " ".join((content or "").split())[:1400]
        sources.append(f"[C{chunk_id} p{page}] {snippet}")
    sources_block = "\n\n".join(sources)

    return f"""You are a research assistant. Answer the question using ONLY the sources below.
If the sources do not contain the answer, say: "I don't have enough information in the provided sources."

Question: {query}

Sources:
{sources_block}

OUTPUT RULES (follow exactly):
- Output ONLY a numbered list: 1) 2) 3) ...
- Write 3–6 items.
- Each item MUST be EXACTLY ONE sentence (one claim).
- Each item MUST end with exactly ONE citation: [C<chunk_id> p<page>]
- Do NOT include quotes.
- Do NOT include extra text before or after the numbered list.
- Do NOT invent facts not present in the sources.
Example:
1) The dynamic call graph includes only the calls observed during a particular execution. [C11 p2]
2) The static call graph is derived from examining the program/object code for possible calls. [C27 p5]
"""


def ollama_generate(prompt: str) -> str:
    """Call Ollama local model (non-stream)."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")


def validate_citations_only(answer: str, hits) -> tuple[bool, str]:
    """
    Validation guarantees:
    - Has at least 2 citations
    - Every citation refers to a retrieved chunk_id
    - Citation page matches the retrieved chunk's page
    - Output is a numbered list (basic format check)
    """
    text = (answer or "").strip()
    if not text:
        return False, "Empty answer."

    # Basic format: must start with "1)"
    if not re.match(r"^1\)\s", text):
        return False, "Answer is not a numbered list starting with '1)'."

    allowed = {str(chunk_id): int(page) for chunk_id, page, content, dist in hits}

    citations = CITE_RE.findall(text)
    if len(citations) < 2:
        return False, f"Too few citations ({len(citations)})."

    for chunk_id, page_str in citations:
        if chunk_id not in allowed:
            return False, f"Cites chunk C{chunk_id} which wasn't retrieved."
        if int(page_str) != allowed[chunk_id]:
            return False, f"Page mismatch for C{chunk_id}: cited p{page_str}, actual p{allowed[chunk_id]}."

    # Enforce: each numbered item ends with one citation
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    item_lines = [ln for ln in lines if re.match(r"^\d+\)\s", ln)]
    if not item_lines:
        return False, "No numbered items found."

    for ln in item_lines:
        # exactly one citation and it's at the end
        matches = CITE_RE.findall(ln)
        if len(matches) != 1:
            return False, "Each item must contain exactly one citation."
        if not re.search(r"\[C\d+\s+p\d+\]\s*$", ln):
            return False, "Each item must end with a citation."

    return True, "OK"


def rewrite_to_template(bad_answer: str, query: str, hits, fail_reason: str) -> str:
    """Ask the model to rewrite ONLY for formatting/grounding (no new facts)."""
    sources = []
    for chunk_id, page, content, dist in hits:
        snippet = " ".join((content or "").split())[:1400]
        sources.append(f"[C{chunk_id} p{page}] {snippet}")
    sources_block = "\n\n".join(sources)

    return ollama_generate(
        f"""Rewrite the answer to follow the OUTPUT RULES exactly.

Validation failure reason: {fail_reason}

Question: {query}

Sources (use ONLY these):
{sources_block}

Bad answer to rewrite (do NOT add new info):
{bad_answer}

OUTPUT RULES (follow exactly):
- Output ONLY a numbered list: 1) 2) 3) ...
- Write 3–6 items.
- Each item MUST be EXACTLY ONE sentence.
- Each item MUST end with exactly ONE citation: [C<chunk_id> p<page>]
- Do NOT include quotes.
- Do NOT include extra text before or after the numbered list.
- Do NOT invent facts.
"""
    )


# -----------------------------
# Main
# -----------------------------
def main():
    paper_id = int(os.environ.get("PAPER_ID", "4"))
    query = os.environ.get("QUERY", "What is the difference between static and dynamic call graphs?")
    top_k = int(os.environ.get("TOP_K", "15"))

    hits = retrieve(paper_id, query, top_k=top_k)

    # Filter junk and keep best few for prompting
    hits = [h for h in hits if not is_junk(h[2])]
    hits = hits[:10]  # keep prompt small/clean

    print("\nTop retrieved sources:")
    for chunk_id, page, content, dist in hits[:5]:
        preview = " ".join((content or "").split())[:120]
        print(f"- C{chunk_id} p{page} dist={dist:.4f}: {preview}...")

    prompt = build_prompt(query, hits)
    answer = ollama_generate(prompt)

    # Retry loop: generate -> validate -> rewrite up to 3 times
    for attempt in range(3):
        ok, msg = validate_citations_only(answer, hits)
        if ok:
            break
        print(f"\n⚠️ Validation failed (attempt {attempt+1}/3): {msg}")
        answer = rewrite_to_template(answer, query, hits, msg)

    ok, msg = validate_citations_only(answer, hits)
    if not ok:
        print("\n❌ Still failed after retries:", msg)
        print("Refusing answer (not grounded).")
        print("\n--- Model output (debug) ---\n")
        print(answer)
        return

    print("\n=== Answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()