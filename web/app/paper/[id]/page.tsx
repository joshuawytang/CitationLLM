"use client";

import { useMemo, useState } from "react";
import { useParams } from "next/navigation";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE!;

type Source = {
  chunk_id: number;
  page: number;
  score: number; // cosine distance
  text: string;
};

export default function PaperPage() {
  const params = useParams();
  const paperId = useMemo(() => Number(params.id), [params]);

  const [query, setQuery] = useState(
    "What is the difference between static and dynamic call graphs?"
  );
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState<string>("");
  const [sources, setSources] = useState<Source[]>([]);
  const [error, setError] = useState<string | null>(null);

  async function ask() {
    setLoading(true);
    setError(null);
    setAnswer("");
    setSources([]);

    const res = await fetch(`${API_BASE}/answer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paper_id: paperId, query, top_k: 15 }),
    });

    setLoading(false);

    if (!res.ok) {
      const text = await res.text();
      setError(`Answer failed: ${text}`);
      return;
    }

    const data = await res.json();
    setAnswer(data.answer);
    setSources(data.sources);
  }

  return (
    <main className="max-w-5xl mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm text-gray-600">Paper</div>
          <h1 className="text-xl font-semibold">paper_id = {paperId}</h1>
        </div>
        <a className="underline" href="/">
          Back
        </a>
      </div>

      <section className="p-4 border rounded-lg space-y-3">
        <div className="font-medium">Ask a question</div>
        <textarea
          className="w-full border rounded-md p-2"
          rows={3}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button
          className="px-4 py-2 rounded-md border"
          onClick={ask}
          disabled={loading || !query.trim()}
        >
          {loading ? "Thinking..." : "Ask"}
        </button>
        {error && <div className="text-red-600 text-sm">{error}</div>}
      </section>

      <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 border rounded-lg space-y-2">
          <div className="font-medium">Answer</div>
          <pre className="whitespace-pre-wrap text-sm">{answer || "—"}</pre>
        </div>

        <div className="p-4 border rounded-lg space-y-3">
          <div className="font-medium">Sources</div>
          <div className="space-y-2">
            {sources.map((s) => (
              <details key={s.chunk_id} className="border rounded-md p-2">
                <summary className="cursor-pointer text-sm">
                  Chunk C{s.chunk_id} • p{s.page} • dist={s.score.toFixed(4)}
                </summary>
                <div className="text-sm mt-2 whitespace-pre-wrap">
                  {s.text}
                </div>
              </details>
            ))}
            {sources.length === 0 && (
              <div className="text-sm text-gray-600">—</div>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}