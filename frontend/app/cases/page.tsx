"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, Case } from "@/lib/api";
import { ThreeDCard } from "@/components/ThreeDCard";
import { 
  FolderKanban, 
  Loader2, 
  Search, 
  PlusCircle, 
  ChevronRight,
  Activity,
  Zap,
  ShieldAlert,
  Clock,
  Sparkles
} from "lucide-react";

export default function CasesIndex() {
  const [cases, setCases] = useState<Case[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("ALL");

  useEffect(() => {
    api.listCases(100)
      .then(setCases)
      .catch((err) => setError(err.message || "Failed to load cases."))
      .finally(() => setLoading(false));
  }, []);

  const filteredCases = cases.filter((c) => {
    const matchesSearch =
      c.case_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (c.visible_injury?.finding || "").toLowerCase().includes(searchQuery.toLowerCase()) ||
      (c.rule_derived_category || "").toLowerCase().includes(searchQuery.toLowerCase());

    if (statusFilter === "ALL") return matchesSearch;
    if (statusFilter === "ANALYZED") return matchesSearch && c.status === "analyzed";
    if (statusFilter === "HIGH") return matchesSearch && c.rule_derived_category === "HIGH";
    if (statusFilter === "DEMO") return matchesSearch && (c.is_demo || c.sensor_summary?.source_type === "demo");
    return matchesSearch;
  });

  return (
    <div className="space-y-8 flex flex-col flex-1 pb-12">
      {/* Header & New Assessment Button */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white flex items-center gap-2.5">
            <Sparkles className="h-7 w-7 text-cyan-400" />
            <span>Research Cases Archive</span>
            <span className="px-3 py-1 rounded-full text-xs font-bold bg-cyan-950 text-cyan-400 border border-cyan-800">
              {cases.length} TOTAL
            </span>
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Multimodal injury assessment cases stored in local MongoDB repository
          </p>
        </div>

        <Link
          href="/create-case"
          className="px-5 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white font-bold text-xs rounded-xl flex items-center justify-center space-x-2 transition-all shadow-lg shadow-blue-500/25 active:scale-95"
        >
          <PlusCircle className="h-4 w-4 shrink-0" />
          <span>New Assessment</span>
        </Link>
      </div>

      {/* Filter & Search Ribbon */}
      <ThreeDCard glowColor="rgba(56, 189, 248, 0.15)" className="p-4">
        <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4">
          {/* Search Bar */}
          <div className="relative flex-1">
            <Search className="absolute left-3.5 top-3 h-4 w-4 text-slate-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by Case ID, Finding, or Triage Level..."
              className="w-full bg-slate-950/80 border border-slate-800 rounded-xl pl-10 pr-4 py-2.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition-colors font-mono"
            />
          </div>

          {/* Filter Pills */}
          <div className="flex items-center gap-2 overflow-x-auto text-xs font-semibold">
            {[
              { id: "ALL", label: "All Cases" },
              { id: "ANALYZED", label: "Analyzed" },
              { id: "HIGH", label: "High Risk" },
              { id: "DEMO", label: "Demo Runs" },
            ].map((f) => (
              <button
                key={f.id}
                onClick={() => setStatusFilter(f.id)}
                className={`px-3.5 py-2 rounded-xl transition-all ${
                  statusFilter === f.id
                    ? "bg-cyan-600 text-white font-bold shadow-md shadow-cyan-600/30"
                    : "bg-slate-950/80 text-slate-400 hover:text-white hover:bg-slate-800 border border-slate-800"
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>
      </ThreeDCard>

      {/* Loading & Error States */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-16 space-y-3">
          <Loader2 className="h-8 w-8 text-cyan-400 animate-spin" />
          <span className="text-xs text-slate-400 font-mono">Loading research cases from database...</span>
        </div>
      )}

      {error && (
        <div className="p-4 bg-rose-950/30 border border-rose-900/60 text-rose-200 rounded-2xl text-xs flex items-center gap-2 backdrop-blur-md">
          <ShieldAlert className="h-4 w-4 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Cases List Grid */}
      {!loading && !error && filteredCases.length === 0 && (
        <div className="p-12 text-center bg-slate-900/80 border border-slate-800 rounded-2xl space-y-3 backdrop-blur-md">
          <FolderKanban className="h-10 w-10 text-slate-600 mx-auto" />
          <p className="text-sm font-semibold text-slate-300">No cases matched your filter criteria.</p>
          <p className="text-xs text-slate-500">Try adjusting your search query or start a new triage assessment.</p>
        </div>
      )}

      {!loading && !error && filteredCases.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {filteredCases.map((c) => {
            const isDemo = c.is_demo || c.sensor_summary?.source_type === "demo";
            const cat = c.rule_derived_category || c.xgboost_prediction?.class || "PENDING";
            const finding = c.visible_injury?.finding || "Injury Assessment";
            const conf = c.visible_injury?.confidence;

            const glow = cat === "HIGH" ? "rgba(244, 63, 94, 0.25)" : (cat === "MODERATE" ? "rgba(245, 158, 11, 0.25)" : "rgba(16, 185, 129, 0.25)");

            return (
              <ThreeDCard key={c.case_id} glowColor={glow} className="p-5 space-y-3">
                <Link href={`/cases/${c.case_id}`} className="block space-y-3 group h-full">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-bold text-cyan-400 bg-cyan-950/60 px-2.5 py-1 rounded-lg border border-cyan-800/50">
                      {c.case_id.slice(0, 14)}...
                    </span>
                    <span
                      className={`px-2.5 py-1 rounded text-[10px] font-bold uppercase tracking-wider ${
                        cat === "HIGH"
                          ? "bg-rose-950 text-rose-400 border border-rose-800"
                          : cat === "MODERATE"
                          ? "bg-amber-950 text-amber-400 border border-amber-800"
                          : "bg-emerald-950 text-emerald-400 border border-emerald-800"
                      }`}
                    >
                      {cat}
                    </span>
                  </div>

                  <div>
                    <h3 className="font-bold text-base text-white group-hover:text-cyan-400 transition-colors capitalize">
                      {finding}
                    </h3>
                    <p className="text-xs text-slate-400 mt-1 flex items-center gap-2 font-mono">
                      <span>Status: {c.status || "created"}</span>
                      {conf != null && (
                        <span className="text-emerald-400 font-bold">({(conf * 100).toFixed(0)}% conf)</span>
                      )}
                    </p>
                  </div>

                  <div className="pt-3 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
                    <div className="flex items-center gap-1.5 font-mono text-[10px]">
                      {isDemo ? (
                        <span className="flex items-center gap-1 text-purple-400 font-bold">
                          <Zap className="h-3 w-3" /> DEMO RUN
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 text-slate-400">
                          <Activity className="h-3 w-3 text-cyan-400" /> MULTIMODAL
                        </span>
                      )}
                    </div>
                    <span className="flex items-center gap-1 text-cyan-400 font-semibold group-hover:translate-x-1 transition-transform">
                      View Details <ChevronRight className="h-3.5 w-3.5" />
                    </span>
                  </div>
                </Link>
              </ThreeDCard>
            );
          })}
        </div>
      )}
    </div>
  );
}
