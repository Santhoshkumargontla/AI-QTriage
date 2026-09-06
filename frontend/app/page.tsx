"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { api, Case } from "@/lib/api";
import { ThreeDCard } from "@/components/ThreeDCard";
import { 
  ShieldAlert, 
  Activity, 
  AlertTriangle, 
  FolderOpen, 
  Loader2, 
  PlusCircle, 
  Zap, 
  AlertOctagon, 
  FlaskConical, 
  BarChart3,
  Sparkles,
  ChevronRight,
  ArrowUpRight,
  CheckCircle2
} from "lucide-react";

export default function Dashboard() {
  const router = useRouter();
  const [cases, setCases] = useState<Case[]>([]);
  const [loading, setLoading] = useState(true);
  const [demoLoading, setDemoLoading] = useState(false);
  const [dbError, setDbError] = useState<string | null>(null);

  const handleOneClickDemo = async () => {
    setDemoLoading(true);
    setDbError(null);
    try {
      const demoCase = await api.runCompleteDemo();
      router.push(`/cases/${demoCase.case_id}`);
    } catch (err: any) {
      console.error(err);
      setDbError(err.message || "Failed to launch E2E demo case.");
    } finally {
      setDemoLoading(false);
    }
  };

  const loadDashboardData = async () => {
    try {
      const health = await api.getHealth();
      if (health.status !== "healthy") {
        throw new Error("Backend health check failed.");
      }
      
      const casesList = await api.listCases(100);
      setCases(casesList);
      setDbError(null);
    } catch (err: any) {
      console.error(err);
      setDbError(
        "MongoDB connection unavailable. Verify MONGODB_URI setting in backend environment."
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDashboardData();

    const onFocus = () => {
      loadDashboardData();
    };
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, []);

  return (
    <div className="space-y-8 flex flex-col flex-1 pb-12">
      {/* Title section */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-3 flex-wrap">
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white flex items-center gap-2.5">
              <Sparkles className="h-7 w-7 text-cyan-400" />
              <span>AI-QTriage Dashboard</span>
            </h1>
            <span className="px-3 py-1 rounded-full text-xs font-bold bg-cyan-950/80 text-cyan-400 border border-cyan-700/60 shadow-lg">
              RESEARCH PROTOTYPE v1.3.0
            </span>
          </div>
          <p className="text-xs sm:text-sm text-slate-400 max-w-2xl">
            Hybrid AI–Quantum Multimodal Framework for Explainable Injury Assessment and Emergency Support
          </p>
        </div>

        {/* Quick Action Header Buttons */}
        <div className="flex items-center gap-3 w-full md:w-auto">
          <Link
            href="/create-case"
            className="flex-1 md:flex-initial px-5 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white font-bold text-xs rounded-xl flex items-center justify-center space-x-2 transition-all shadow-lg shadow-blue-500/25 active:scale-95"
          >
            <PlusCircle className="h-4 w-4 shrink-0" />
            <span>New Assessment</span>
          </Link>
          <button
            onClick={handleOneClickDemo}
            disabled={demoLoading || loading}
            className="flex-1 md:flex-initial px-5 py-2.5 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-bold text-xs rounded-xl flex items-center justify-center space-x-2 transition-all shadow-lg shadow-purple-500/25 active:scale-95 disabled:opacity-50"
          >
            {demoLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4 shrink-0" />}
            <span>Run Complete Demo</span>
          </button>
        </div>
      </div>

      {/* Mandatory Medical Safety Disclaimer */}
      <div className="p-4 bg-amber-950/30 border border-amber-900/60 rounded-2xl flex items-start space-x-3 text-amber-300 text-xs backdrop-blur-md">
        <ShieldAlert className="h-5 w-5 flex-shrink-0 mt-0.5 text-amber-400" />
        <div className="space-y-1">
          <span className="font-bold block text-amber-300">RESEARCH PROTOTYPE MEDICAL SAFETY DISCLAIMER</span>
          <p className="leading-relaxed text-amber-200/80">
            AI-QTriage is an experimental research framework for academic evaluation. It does not diagnose medical conditions or recommend therapy. An ordinary photograph cannot reliably identify fractures or internal injuries. Experimental model categories are not clinical triage decisions.
          </p>
        </div>
      </div>

      {dbError && (
        <div className="p-4 bg-rose-950/40 border border-rose-900/80 rounded-2xl text-rose-200 text-xs space-y-2 backdrop-blur-md">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="h-4 w-4 text-rose-400" />
            <h3 className="font-semibold text-sm">Database Connection Notice</h3>
          </div>
          <p>{dbError}</p>
        </div>
      )}

      {/* TOP STAT CARDS GRID (3D Tilt & Glow) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {/* Total Cases */}
        <ThreeDCard glowColor="rgba(56, 189, 248, 0.25)" className="p-5">
          <div className="space-y-3">
            <div className="flex items-center justify-between text-slate-400 text-xs">
              <span className="font-semibold uppercase tracking-wider">Total Cases</span>
              <div className="p-2 bg-blue-950/60 rounded-xl border border-blue-800/40">
                <FolderOpen className="h-4 w-4 text-cyan-400" />
              </div>
            </div>
            <div className="text-3xl font-black text-white">{cases.length}</div>
            <div className="text-[11px] text-cyan-400 font-mono flex items-center gap-1">
              <CheckCircle2 className="h-3 w-3" /> Indexed in MongoDB
            </div>
          </div>
        </ThreeDCard>

        {/* Completed Analyses */}
        <ThreeDCard glowColor="rgba(16, 185, 129, 0.25)" className="p-5">
          <div className="space-y-3">
            <div className="flex items-center justify-between text-slate-400 text-xs">
              <span className="font-semibold uppercase tracking-wider">Completed Analyses</span>
              <div className="p-2 bg-emerald-950/60 rounded-xl border border-emerald-800/40">
                <Activity className="h-4 w-4 text-emerald-400" />
              </div>
            </div>
            <div className="text-3xl font-black text-emerald-400">
              {cases.filter(c => c.status === "analyzed").length}
            </div>
            <div className="text-[11px] text-emerald-400/90 font-mono">
              Vision + Multimodal Fusion
            </div>
          </div>
        </ThreeDCard>

        {/* Research Experiments */}
        <ThreeDCard glowColor="rgba(168, 85, 247, 0.25)" className="p-5">
          <div className="space-y-3">
            <div className="flex items-center justify-between text-slate-400 text-xs">
              <span className="font-semibold uppercase tracking-wider">Quantum Experiments</span>
              <div className="p-2 bg-purple-950/60 rounded-xl border border-purple-800/40">
                <FlaskConical className="h-4 w-4 text-purple-400" />
              </div>
            </div>
            <div className="text-3xl font-black text-purple-400">
              {cases.filter(c => c.quantum_prediction).length}
            </div>
            <div className="text-[11px] text-purple-400/90 font-mono">
              4-Qubit VQC Outputs Stored
            </div>
          </div>
        </ThreeDCard>

        {/* Demo Runs */}
        <ThreeDCard glowColor="rgba(245, 158, 11, 0.25)" className="p-5">
          <div className="space-y-3">
            <div className="flex items-center justify-between text-slate-400 text-xs">
              <span className="font-semibold uppercase tracking-wider">Demo Runs</span>
              <div className="p-2 bg-amber-950/60 rounded-xl border border-amber-800/40">
                <Zap className="h-4 w-4 text-amber-400" />
              </div>
            </div>
            <div className="text-3xl font-black text-amber-400">
              {cases.filter(c => c.is_demo || c.sensor_summary?.source_type === "demo").length}
            </div>
            <div className="text-[11px] text-amber-400/90 font-mono">
              Synthetic Football Fall Data
            </div>
          </div>
        </ThreeDCard>
      </div>

      {/* QUICK ACTIONS ROW (3D Cards) */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <Zap className="h-5 w-5 text-cyan-400" />
            Quick Research Actions
          </h2>
          <span className="text-xs text-slate-400 font-mono">Interactive Launchpad</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 text-xs">
          <ThreeDCard glowColor="rgba(56, 189, 248, 0.2)" className="p-5">
            <Link href="/create-case" className="block space-y-3 group h-full">
              <div className="flex justify-between items-start">
                <div className="p-2.5 bg-cyan-950/70 border border-cyan-800/50 rounded-xl">
                  <PlusCircle className="h-5 w-5 text-cyan-400 group-hover:scale-110 transition-transform" />
                </div>
                <ArrowUpRight className="h-4 w-4 text-slate-500 group-hover:text-cyan-400 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
              </div>
              <div>
                <span className="font-bold text-white text-sm block">New Assessment</span>
                <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
                  Upload injury photo, fill questionnaire, and execute triage analysis.
                </p>
              </div>
            </Link>
          </ThreeDCard>

          <ThreeDCard glowColor="rgba(168, 85, 247, 0.2)" className="p-5">
            <button
              type="button"
              onClick={handleOneClickDemo}
              disabled={demoLoading || loading}
              className="w-full text-left space-y-3 group h-full disabled:opacity-50"
            >
              <div className="flex justify-between items-start">
                <div className="p-2.5 bg-purple-950/70 border border-purple-800/50 rounded-xl">
                  {demoLoading ? <Loader2 className="h-5 w-5 text-purple-400 animate-spin" /> : <Zap className="h-5 w-5 text-purple-400 group-hover:scale-110 transition-transform" />}
                </div>
                <ArrowUpRight className="h-4 w-4 text-slate-500 group-hover:text-purple-400 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
              </div>
              <div>
                <span className="font-bold text-white text-sm block">Run Complete Demo</span>
                <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
                  Instantly launch synthetic football fall case with all model predictions.
                </p>
              </div>
            </button>
          </ThreeDCard>

          <ThreeDCard glowColor="rgba(244, 63, 94, 0.2)" className="p-5">
            <Link href="/cases" className="block space-y-3 group h-full">
              <div className="flex justify-between items-start">
                <div className="p-2.5 bg-rose-950/70 border border-rose-800/50 rounded-xl">
                  <AlertOctagon className="h-5 w-5 text-rose-400 group-hover:scale-110 transition-transform" />
                </div>
                <ArrowUpRight className="h-4 w-4 text-slate-500 group-hover:text-rose-400 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
              </div>
              <div>
                <span className="font-bold text-white text-sm block">Test SOS Simulation</span>
                <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
                  Open an existing case to test live Twilio SOS SMS alert dispatch.
                </p>
              </div>
            </Link>
          </ThreeDCard>

          <ThreeDCard glowColor="rgba(16, 185, 129, 0.2)" className="p-5">
            <Link href="/research" className="block space-y-3 group h-full">
              <div className="flex justify-between items-start">
                <div className="p-2.5 bg-emerald-950/70 border border-emerald-800/50 rounded-xl">
                  <BarChart3 className="h-5 w-5 text-emerald-400 group-hover:scale-110 transition-transform" />
                </div>
                <ArrowUpRight className="h-4 w-4 text-slate-500 group-hover:text-emerald-400 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
              </div>
              <div>
                <span className="font-bold text-white text-sm block">Research Benchmarks</span>
                <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
                  View model registry SHA-256 hashes, ablation, and QML metrics.
                </p>
              </div>
            </Link>
          </ThreeDCard>
        </div>
      </div>

      {/* RECENT CASES TABLE (3D Card Container) */}
      <ThreeDCard glowColor="rgba(56, 189, 248, 0.15)" className="p-6 space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-slate-800 pb-4 gap-2">
          <div className="flex items-center space-x-2.5">
            <FolderOpen className="h-5 w-5 text-cyan-400 shrink-0" />
            <h2 className="text-base font-bold text-white">Recent Research Cases</h2>
          </div>
          <Link href="/create-case" className="text-xs text-cyan-400 hover:text-cyan-300 font-semibold shrink-0 flex items-center gap-1">
            + New Assessment <ChevronRight className="h-3.5 w-3.5" />
          </Link>
        </div>

        {loading ? (
          <div className="flex flex-col items-center justify-center py-12 space-y-2">
            <Loader2 className="h-6 w-6 text-cyan-400 animate-spin" />
            <span className="text-slate-400 text-xs font-mono">Loading recent cases from MongoDB...</span>
          </div>
        ) : cases.length === 0 ? (
          <div className="text-slate-400 text-xs py-12 text-center bg-slate-950/50 border border-slate-800 rounded-xl space-y-2">
            <FolderOpen className="h-8 w-8 text-slate-600 mx-auto" />
            <p className="font-semibold text-slate-300">No research cases found in database.</p>
            <p className="text-slate-500">Initialize a new assessment or click "Run Complete Demo" above.</p>
          </div>
        ) : (
          <>
            {/* Mobile Cards */}
            <div className="sm:hidden space-y-3">
              {cases.slice(0, 10).map((c) => (
                <Link
                  key={c.case_id}
                  href={`/cases/${c.case_id}`}
                  className="block p-4 rounded-xl border border-slate-800 bg-slate-950/60 space-y-2 hover:border-cyan-500/40 transition-colors"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-xs font-bold text-cyan-400">
                      {c.case_id.substring(0, 10)}...
                    </span>
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                      c.status === "analyzed"
                        ? "bg-emerald-950 text-emerald-400 border border-emerald-800"
                        : "bg-amber-950 text-amber-400 border border-amber-800"
                    }`}>
                      {c.status}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 font-mono">
                    {new Date(c.created_at).toLocaleString()}
                  </p>
                  <p className="text-xs text-slate-300 font-semibold">
                    {c.image_reference && c.questionnaire && c.sensor_summary
                      ? "FULL MULTIMODAL FUSION"
                      : "REDUCED MODALITY MODE"}
                  </p>
                </Link>
              ))}
            </div>

            {/* Desktop Table */}
            <div className="hidden sm:block overflow-x-auto border border-slate-800 rounded-xl">
              <table className="w-full text-xs text-left text-slate-300">
                <thead className="text-[11px] text-slate-400 font-semibold uppercase bg-slate-950 border-b border-slate-800">
                  <tr>
                    <th scope="col" className="py-3 px-4">Case ID</th>
                    <th scope="col" className="py-3 px-4">Created At</th>
                    <th scope="col" className="py-3 px-4">Status</th>
                    <th scope="col" className="py-3 px-4">Modality Configuration</th>
                    <th scope="col" className="py-3 px-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 font-mono text-[11px]">
                  {cases.slice(0, 10).map((c) => (
                    <tr key={c.case_id} className="hover:bg-slate-800/40 transition-colors">
                      <td className="py-3 px-4 font-bold text-cyan-400">{c.case_id.substring(0, 12)}...</td>
                      <td className="py-3 px-4 text-slate-400 font-sans">
                        {new Date(c.created_at).toLocaleString()}
                      </td>
                      <td className="py-3 px-4 font-sans">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          c.status === "analyzed" 
                            ? "bg-emerald-950 text-emerald-400 border border-emerald-800" 
                            : "bg-amber-950 text-amber-400 border border-amber-800"
                        }`}>
                          {c.status}
                        </span>
                      </td>
                      <td className="py-3 px-4 font-sans text-slate-300 font-medium">
                        {c.image_reference && c.questionnaire && c.sensor_summary 
                          ? "FULL MULTIMODAL FUSION" 
                          : "REDUCED MODALITY MODE"}
                      </td>
                      <td className="py-3 px-4 text-right font-sans">
                        <Link 
                          href={`/cases/${c.case_id}`}
                          className="px-3 py-1 bg-cyan-950 hover:bg-cyan-900 text-cyan-300 border border-cyan-800 rounded-lg font-semibold text-xs transition-all inline-flex items-center gap-1"
                        >
                          Open Case <ChevronRight className="h-3 w-3" />
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </ThreeDCard>
    </div>
  );
}
