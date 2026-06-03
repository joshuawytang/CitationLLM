# CitationLLM — LitReview Copilot

A local-first research assistant that answers questions about your PDFs with **verifiable, page-level citations**. Upload a paper, ask a question, and get a numbered list of one-sentence claims — each ending in a `[C<chunk> p<page>]` citation that is validated against the actual retrieved source before it's ever shown to you.

The goal is **grounding**: the model is only allowed to answer from retrieved passages, and every citation is machine-checked (correct chunk, correct page) with an automatic rewrite loop if it fails.

## How it works

```
PDF ──▶ pypdf extract ──▶ line-chunk ──▶ MiniLM embed ──▶ Postgres/pgvector
                                                                │
question ──▶ embed ──▶ cosine kNN retrieve ──▶ prompt ──▶ Ollama (llama3.1)
                                                                │
                                              validate citations ──▶ rewrite ×3 ──▶ answer
```

1. **Ingest** — each PDF page is split into overlapping line-chunks, embedded with `all-MiniLM-L6-v2` (384-dim), and stored in Postgres with the [pgvector](https://github.com/pgvector/pgvector) extension.
2. **Retrieve** — the question is embedded and the top-k nearest chunks (cosine distance) for that paper are fetched. Junk chunks (too short / low letter ratio) are filtered out.
3. **Generate** — a local [Ollama](https://ollama.com) model answers using *only* the retrieved chunks, under strict output rules (numbered list, one claim per line, exactly one citation per line).
4. **Validate** — every citation is checked against the retrieved set: the chunk must have been retrieved and the cited page must match. On failure, the model is asked to rewrite (up to 3 attempts) before the request errors out.

## Repo layout

| Path | What it is |
|------|------------|
| `services/api/` | FastAPI backend — ingestion, retrieval, RAG, citation validation |
| `web/` | Next.js 16 / React 19 frontend (upload papers, ask questions) |
| `infra/` | `docker-compose.yml` for the pgvector Postgres database |
| `apps/`, `packages/` | Reserved for future workspace apps / shared code |

Key backend files:
- [`services/api/main.py`](services/api/main.py) — the FastAPI app and all routes
- [`services/api/ingest_pdf.py`](services/api/ingest_pdf.py) — standalone PDF ingestion script
- [`services/api/rag_answer.py`](services/api/rag_answer.py) / [`search_local.py`](services/api/search_local.py) — CLI helpers for RAG / vector search

## Prerequisites

- **Python 3.11+**
- **Node.js 20+**
- **Docker** (for Postgres + pgvector)
- **[Ollama](https://ollama.com)** running locally with a model pulled:
  ```bash
  ollama pull llama3.1:8b
  ```

## Getting started

### 1. Start the database

```bash
docker compose -f infra/docker-compose.yml up -d
```

This starts Postgres 16 with pgvector on `localhost:5432` (db `litreview`, user/password `postgres`/`postgres`).

### 2. Create the schema

There's no migration tool yet, so create the extension and tables once:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE papers (
  id         SERIAL PRIMARY KEY,
  title      TEXT,
  filename   TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
  id         SERIAL PRIMARY KEY,
  paper_id   INTEGER NOT NULL REFERENCES papers(id),
  page_start INTEGER,
  page_end   INTEGER,
  section    TEXT,
  content    TEXT,
  embedding  vector(384)
);
```

### 3. Run the API

```bash
cd services/api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The first run downloads the MiniLM embedding model. Health check: `curl localhost:8000/health`.

### 4. Run the web app

```bash
cd web
npm install
echo "NEXT_PUBLIC_API_BASE=http://localhost:8000" > .env.local
npm run dev
```

Open [http://localhost:3000](http://localhost:3000), upload a PDF, then open it and ask a question.

## Configuration

The backend is configured via environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/litreview` | Postgres connection string |
| `DATA_DIR` | `data` | Where uploaded PDFs are stored |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model used for generation |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama generate endpoint |

Frontend: `NEXT_PUBLIC_API_BASE` (in `web/.env.local`) points to the API.

## API reference

| Method | Route | Description |
|--------|-------|-------------|
| `GET`  | `/health` | Liveness check |
| `GET`  | `/papers` | List ingested papers |
| `POST` | `/papers/upload` | Upload + index a PDF (multipart `file`) |
| `POST` | `/search` | Vector search within a paper (`{paper_id, query, top_k}`) |
| `POST` | `/answer` | Grounded RAG answer with validated citations (`{paper_id, query, top_k}`) |

Example:

```bash
curl -s localhost:8000/answer \
  -H 'Content-Type: application/json' \
  -d '{"paper_id": 1, "query": "What is the difference between static and dynamic call graphs?"}'
```

## Notes

- Everything runs **locally** — embeddings (MiniLM) and generation (Ollama) require no external API keys.
- Citation format is `[C<chunk_id> p<page>]`; the validator in [`main.py`](services/api/main.py) rejects answers that cite unretrieved chunks or mismatched pages.
