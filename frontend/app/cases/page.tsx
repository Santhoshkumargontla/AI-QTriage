"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, Case } from "@/lib/api";
import { FolderKanban, Loader2 } from "lucide-react";

export default function CasesIndex() {
  const [cases, setCases] = useState<Case[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listCases(50)
      .then(setCases)
      .catch((err) => setError(err.message || "Failed to load cases."))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-extrabold text-[var(--text-main)] flex items-center gap-2">
          <FolderKanban className="h-6 w-6 text-blue-400" />
          My Cases
        </h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">Research assessments stored in MongoDB. Not a clinical record.</p>
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-slate-400 text-sm">
          <Loader2 className="h-4 w-4 animate-spin" /> Loading cases…
        </div>
      )}
      {error && <p className="text-sm text-red-400">{error}</p>}
      {!loading && !error && cases.length === 0 && (
        <p className="text-sm text-slate-400">No cases yet. Start a new assessment.</p>
      )}
      <ul className="space-y-2">
        {cases.map((c) => (
          <li key={c.case_id}>
            <Link
              href={`/cases/${c.case_id}`}
              className="block p-4 rounded-xl border border-[var(--border-card)] bg-[var(--bg-card)] hover:border-blue-500 text-base text-[var(--text-main)]"
            >
              <span className="font-mono text-sm text-[var(--text-muted)] break-all">{c.case_id}</span>
              <span className="block text-[var(--text-sub)] mt-1">{c.status || "unknown status"}</span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
