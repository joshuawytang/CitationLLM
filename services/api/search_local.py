import os
import psycopg
from sentence_transformers import SentenceTransformer

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/litreview",
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def embed(text: str) -> list[float]:
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()

def main():
    paper_id = int(os.environ.get("PAPER_ID", "3"))  # default to your ingested paper_id=3
    query = os.environ.get("QUERY", "What skills and technologies are mentioned?")

    qvec = embed(query)

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            # NOTE: <=> is cosine distance; smaller = more similar
            cur.execute(
                """
                SELECT page_start, section, content,
                       (embedding <=> %s::vector) AS cosine_distance
                FROM chunks
                WHERE paper_id = %s
                ORDER BY cosine_distance ASC
                LIMIT 5
                """,
                (qvec, paper_id),
            )
            rows = cur.fetchall()

    print(f"\nPaper ID: {paper_id}")
    print(f"Query: {query}\n")
    for i, (page, section, content, dist) in enumerate(rows, start=1):
        snippet = " ".join(content.split())[:220]
        print(f"{i}. p{page} [{section}] dist={dist:.4f}")
        print(f"   {snippet}")
        print()

if __name__ == "__main__":
    main()