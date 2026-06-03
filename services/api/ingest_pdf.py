import os
from pathlib import Path
import psycopg
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/litreview")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims
model = SentenceTransformer(MODEL_NAME)

def embed(text: str) -> list[float]:
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    text = " ".join(text.split())  # normalize whitespace
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        i += max(1, chunk_size - overlap)
    return chunks

def ingest_pdf(pdf_path: Path, title: str | None = None):
    reader = PdfReader(str(pdf_path))

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            # 1) Insert paper row
            cur.execute(
                "INSERT INTO papers (title, filename) VALUES (%s, %s) RETURNING id",
                (title or pdf_path.stem, pdf_path.name),
            )
            paper_id = cur.fetchone()[0]

            # 2) For each page: extract text → chunk → embed → insert
            for page_idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                page_text = page_text.strip()
                if not page_text:
                    continue

                for chunk in chunk_text(page_text):
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

    print(f"Ingested {pdf_path.name} as paper_id={paper_id}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest_pdf.py path/to/paper.pdf")
        raise SystemExit(1)

    ingest_pdf(Path(sys.argv[1]))