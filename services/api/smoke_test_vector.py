import os
import psycopg
from openai import OpenAI

DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/litreview")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)

def embed(text: str) -> list[float]:
    # 768 dims as chosen
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        dimensions=768,
    )
    return resp.data[0].embedding

def main():
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            # Insert a paper
            cur.execute(
                "INSERT INTO papers (title, filename) VALUES (%s, %s) RETURNING id",
                ("Smoke Test Paper", "smoke.pdf"),
            )
            paper_id = cur.fetchone()[0]

            docs = [
                ("Abstract", 1, 1, "This paper proposes a transformer method for retrieval."),
                ("Methods", 2, 2, "We use embeddings and cosine similarity for semantic search."),
                ("Results", 3, 3, "Results show improved accuracy on benchmarks."),
            ]

            for section, p1, p2, text in docs:
                vec = embed(text)
                cur.execute(
                    """
                    INSERT INTO chunks (paper_id, page_start, page_end, section, content, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                    """,
                    (paper_id, p1, p2, section, text, vec),
                )

            conn.commit()

            # Optional but recommended once you’ve inserted data:
            cur.execute("ANALYZE chunks;")
            conn.commit()

            query = "How does the paper do semantic search?"
            qvec = embed(query)

            cur.execute(
                """
                SELECT section, page_start, content,
                       (embedding <=> %s::vector) AS cosine_distance
                FROM chunks
                ORDER BY cosine_distance ASC
                LIMIT 3
                """,
                (qvec,),
            )

            print(f"\nQuery: {query}\nTop matches:")
            for section, page, content, dist in cur.fetchall():
                print(f"- [{section} p{page}] dist={dist:.4f} :: {content}")

if __name__ == "__main__":
    main()