import os
import re
import uuid
from pathlib import Path
from typing import Optional

import psycopg
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
import requests
from sentence_transformers import SentenceTransformer



# -----------------------------
# Config
# -----------------------------
DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/litreview",
)

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims
embedder = SentenceTransformer(EMBED_MODEL_NAME)

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

CITE_RE = re.compile(r"\[C(\d+)\s+p(\d+)\]")

app = FastAPI(title="LitReview Copilot API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# DB helpers
# -----------------------------
def db_connect():
    return psycopg.connect(DB_URL)


def embed(text: str) -> list[float]:
    vec = embedder.encode(text, normalize_embeddings=True)
    return vec.tolist()


def is_junk(text: str) -> bool:
    t = " ".join((text or "").split())
    if len(t) < 120:
        return True
    letters = sum(ch.isalpha() for ch in t)
    return (letters / max(1, len(t))) < 0.45


def chunk_text_lines(text: str, lines_per_chunk: int = 10, overlap_lines: int = 2) -> list[str]:
    """
    Better chunking than raw char windows for many PDFs.
    Splits by lines and groups them.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    chunks = []
    i = 0
    step = max(1, lines_per_chunk - overlap_lines)
    while i < len(lines):
        chunk = " ".join(lines[i:i + lines_per_chunk])
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks


# -----------------------------
# Ingestion
# -----------------------------
def ingest_pdf_to_db(pdf_path: Path, title: Optional[str] = None) -> int:
    reader = PdfReader(str(pdf_path))

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO papers (title, filename) VALUES (%s, %s) RETURNING id",
                (title or pdf_path.stem, pdf_path.name),
            )
            paper_id = cur.fetchone()[0]

            for page_idx, page in enumerate(reader.pages, start=1):
                page_text = (page.extract_text() or "").strip()
                if not page_text:
                    continue

                # chunk this page
                for chunk in chunk_text_lines(page_text):
                    vec = embed(chunk)
                    cur.execute(
                        """
                        INSERT INTO chunks (paper_id, page_start, page_end, section, content, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s::vector)
                        """,
                        (paper_id, page_idx, page_idx, "Unknown", chunk, vec),
                    )

            conn.commit()
            cur.execute("ANALYZE chunks;")
            conn.commit()

    return paper_id


# -----------------------------
# Retrieval
# -----------------------------
def retrieve(paper_id: int, query: str, top_k: int = 15):
    qvec = embed(query)
    with db_connect() as conn:
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
            return cur.fetchall()  # (chunk_id, page, content, dist)


# -----------------------------
# Generation (RAG)
# -----------------------------
def build_prompt(query: str, hits) -> str:
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
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def validate_citations_only(answer: str, hits) -> tuple[bool, str]:
    text = (answer or "").strip()
    if not text:
        return False, "Empty answer."
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

    # each numbered line ends with exactly one citation
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    item_lines = [ln for ln in lines if re.match(r"^\d+\)\s", ln)]
    if not item_lines:
        return False, "No numbered items found."

    for ln in item_lines:
        matches = CITE_RE.findall(ln)
        if len(matches) != 1:
            return False, "Each item must contain exactly one citation."
        if not re.search(r"\[C\d+\s+p\d+\]\s*$", ln):
            return False, "Each item must end with a citation."

    return True, "OK"


def rewrite_to_template(bad_answer: str, query: str, hits, fail_reason: str) -> str:
    sources = []
    for chunk_id, page, content, dist in hits:
        snippet = " ".join((content or "").split())[:1400]
        sources.append(f"[C{chunk_id} p{page}] {snippet}")
    sources_block = "\n\n".join(sources)

    rewrite_prompt = f"""Rewrite the answer to follow the OUTPUT RULES exactly.

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
    return ollama_generate(rewrite_prompt)


def answer_with_rag(paper_id: int, query: str, top_k: int = 15):
    hits = retrieve(paper_id, query, top_k=top_k)
    hits = [h for h in hits if not is_junk(h[2])]
    hits = hits[:10]

    prompt = build_prompt(query, hits)
    answer = ollama_generate(prompt)

    for _ in range(3):
        ok, msg = validate_citations_only(answer, hits)
        if ok:
            return answer, hits
        answer = rewrite_to_template(answer, query, hits, msg)

    ok, msg = validate_citations_only(answer, hits)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Failed grounding validation: {msg}")

    return answer, hits


# -----------------------------
# API models
# -----------------------------
class SearchRequest(BaseModel):
    paper_id: int
    query: str
    top_k: int = 10


class AnswerRequest(BaseModel):
    paper_id: int
    query: str
    top_k: int = 15


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/papers")
def list_papers():
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, filename, created_at FROM papers ORDER BY id DESC LIMIT 200"
            )
            rows = cur.fetchall()

    return [
        {"id": r[0], "title": r[1], "filename": r[2], "created_at": r[3].isoformat()}
        for r in rows
    ]


@app.post("/papers/upload")
async def upload_paper(file: UploadFile = File(...), title: Optional[str] = None):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")

    # Save file to disk
    safe_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    out_path = DATA_DIR / safe_name
    content = await file.read()
    out_path.write_bytes(content)

    # Ingest into DB
    try:
        paper_id = ingest_pdf_to_db(out_path, title=title)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return {"paper_id": paper_id, "stored_as": str(out_path)}


@app.post("/search")
def search(req: SearchRequest):
    hits = retrieve(req.paper_id, req.query, top_k=req.top_k)
    results = []
    for chunk_id, page, content, dist in hits:
        results.append(
            {
                "chunk_id": chunk_id,
                "page": page,
                "score": float(dist),  # cosine distance; lower is better
                "snippet": " ".join((content or "").split())[:300],
            }
        )
    return {"results": results}


@app.post("/answer")
def answer(req: AnswerRequest):
    answer_text, hits = answer_with_rag(req.paper_id, req.query, top_k=req.top_k)
    sources = [
        {
            "chunk_id": chunk_id,
            "page": page,
            "score": float(dist),
            "text": " ".join((content or "").split())[:1200],
        }
        for chunk_id, page, content, dist in hits
    ]
    return {"answer": answer_text, "sources": sources}