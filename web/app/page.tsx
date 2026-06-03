"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE!;

type Paper = {
  id: number;
  title: string | null;
  filename: string | null;
  created_at: string;
};

export default function HomePage() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadPapers() {
    setError(null);
    const res = await fetch(`${API_BASE}/papers`);
    if (!res.ok) {
      setError(`Failed to load papers: ${res.status}`);
      return;
    }
    const data = await res.json();
    setPapers(data);
  }

  useEffect(() => {
    loadPapers();
  }, []);

  async function upload() {
    if (!file) return;
    setUploading(true);
    setError(null);

    const form = new FormData();
    form.append("file", file);

    const res = await fetch(`${API_BASE}/papers/upload`, {
      method: "POST",
      body: form,
    });

    setUploading(false);

    if (!res.ok) {
      const text = await res.text();
      setError(`Upload failed: ${text}`);
      return;
    }

    // refresh list
    setFile(null);
    await loadPapers();
  }

  return (
    <main className="max-w-3xl mx-auto p-6 space-y-6">
      <h1 className="text-2xl font-semibold">LitReview Copilot</h1>

      <section className="p-4 border rounded-lg space-y-3">
        <div className="font-medium">Upload a PDF</div>
        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <button
          className="px-4 py-2 rounded-md border"
          onClick={upload}
          disabled={!file || uploading}
        >
          {uploading ? "Uploading..." : "Upload & Index"}
        </button>
        {error && <div className="text-red-600 text-sm">{error}</div>}
      </section>

      <section className="space-y-3">
        <div className="font-medium">Your papers</div>
        <div className="space-y-2">
          {papers.map((p) => (
            <Link
              key={p.id}
              href={`/paper/${p.id}`}
              className="block p-3 border rounded-lg hover:bg-gray-50"
            >
              <div className="font-medium">
                {p.title ?? p.filename ?? `Paper ${p.id}`}
              </div>
              <div className="text-sm text-gray-600">
                id={p.id} • {new Date(p.created_at).toLocaleString()}
              </div>
            </Link>
          ))}
          {papers.length === 0 && (
            <div className="text-sm text-gray-600">No papers yet.</div>
          )}
        </div>
      </section>
    </main>
  );
}