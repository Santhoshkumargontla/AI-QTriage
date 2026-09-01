"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { api, Case } from "@/lib/api";
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
  BarChart3 
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

  useEffect(() => {
    async function loadDashboardData() {
      try {
        const health = await api.getHealth();
        if (health.status !== "healthy") {
          throw new Error("Backend health check failed.");
        }
        
        const casesList = await api.listCases(10);
        setCases(casesList);
      } catch (err: any) {
        console.error(err);
        setDbError(
          "MongoDB connection unavailable. Verify MONGODB_URI setting in backend environment."
        );
      } finally {
        setLoading(false);
      }
    }

    loadDashboardData();
  }, []);

  return (
    <div className="space-y-6 flex flex-col flex-1">
      {/* Title section */}
      <div className="flex flex-col gap-4">
        <div className="min-w-0">
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-[var(--text-main)] flex flex-wrap items-center gap-2">
            <span>AI-QTriage Dashboard</span>
            <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-blue-950 text-blue-400 border border-blue-800/40">
              RESEARCH PROTOTYPE
            </span>
          </h1>
          <p className="text-sm text-[var(--text-muted)] mt-1 max-w-2xl">
            Hybrid AI–Quantum Multimodal Framework for Explainable Injury Assessment and Emergency Support
          </p>
        </div>

        {/* Quick Action Header Buttons */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2 w-full sm:w-auto sm:ml-auto sm:self-start">
          <Link
            href="/create-case"
            className="px-4 py-2.5 sm:py-2 bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs rounded-xl flex items-center justify-center space-x-1.5 transition-all shadow-md shadow-blue-600/30"
          >
            <PlusCircle className="h-4 w-4 shrink-0" />
            <span>New Assessment</span>
          </Link>
          <button
            onClick={handleOneClickDemo}
            disabled={demoLoading || loading}
            className="px-4 py-2.5 sm:py-2 bg-purple-600 hover:bg-purple-500 text-white font-bold text-xs rounded-xl flex items-center justify-center space-x-1.5 transition-all shadow-md shadow-purple-600/30 disabled:opacity-50"
          >
            {demoLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4 shrink-0" />}
            <span>Run Complete Demo</span>
          </button>
        </div>
      </div>

      {/* Mandatory Medical Safety Disclaimer */}
      <div className="p-4 bg-amber-950/20 border border-amber-900/50 rounded-2xl flex items-start space-x-3 text-amber-300 text-xs">
        <ShieldAlert className="h-5 w-5 flex-shrink-0 mt-0.5 text-amber-400" />
        <div className="space-y-1">
          <span className="font-bold block">RESEARCH PROTOTYPE MEDICAL SAFETY DISCLAIMER</span>
          <p className="leading-relaxed">
            AI-QTriage is an experimental research framework for academic evaluation. It does not diagnose medical conditions or recommend therapy. An ordinary photograph cannot reliably identify fractures or internal injuries. Experimental model categories are not clinical triage decisions.
          </p>
        </div>
      </div>

      {dbError && (
        <div className="p-4 bg-red-950/30 border border-red-900/60 rounded-2xl text-red-200 text-xs space-y-2">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="h-4 w-4 text-red-400" />
            <h3 className="font-semibold text-sm">Database Connection Notice</h3>
          </div>
          <p>{dbError}</p>
        </div>
      )}

      {/* TOP STAT CARDS GRID (Section 19 requirement) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Cases */}
        <div className="dash-card p-5 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-semibold">Total Cases</span>
            <FolderOpen className="h-4 w-4 text-blue-400" />
          </div>
          <div className="text-2xl font-extrabold text-white">{cases.length}</div>
          <p className="text-[10px] text-slate-500">Indexed in MongoDB</p>
        </div>

        {/* Completed Analyses */}
        <div className="dash-card p-5 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-semibold">Completed Analyses</span>
            <Activity className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-extrabold text-emerald-400">
            {cases.filter(c => c.status === "analyzed").length}
          </div>
          <p className="text-[10px] text-slate-500">Vision + Multimodal Fusion</p>
        </div>

        {/* Research Experiments */}
        <div className="dash-card p-5 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-semibold">Research Experiments</span>
            <FlaskConical className="h-4 w-4 text-purple-400" />
          </div>
          <div className="text-2xl font-extrabold text-purple-400">
            {cases.filter(c => c.quantum_prediction).length}
          </div>
          <p className="text-[10px] text-slate-500">Cases with a stored VQC output (experimental)</p>
        </div>

        {/* Demo Runs */}
        <div className="dash-card p-5 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-semibold">Demo Runs</span>
            <Zap className="h-4 w-4 text-amber-400" />
          </div>
          <div className="text-2xl font-extrabold text-amber-400">
            {cases.filter(c => c.is_demo || c.sensor_summary?.source_type === "demo").length}
          </div>
          <p className="text-[10px] text-slate-500">Synthetic Football Fall Data</p>
        </div>
      </div>

      {/* QUICK ACTIONS ROW */}
      <div className="dash-card p-4 sm:p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[#26324A] pb-3 gap-1">
          <h3 className="font-bold text-white text-base">Quick Research Actions</h3>
          <span className="text-xs text-slate-400">Select an action to launch</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          <Link
            href="/create-case"
            className="p-4 bg-[#0D1426] hover:bg-[#151F35] border border-[#26324A] hover:border-blue-500/40 rounded-xl space-y-2 transition-all block group"
          >
            <PlusCircle className="h-5 w-5 text-blue-400 group-hover:scale-110 transition-transform" />
            <div>
              <span className="font-bold text-white block">New Assessment</span>
              <p className="text-[10px] text-slate-400">Upload photo &amp; run step-by-step triage</p>
            </div>
          </Link>

          <button
            type="button"
            onClick={handleOneClickDemo}
            disabled={demoLoading || loading}
            className="p-4 bg-[#0D1426] hover:bg-[#151F35] border border-[#26324A] hover:border-purple-500/40 rounded-xl space-y-2 text-left transition-all group disabled:opacity-50"
          >
            <Zap className="h-5 w-5 text-purple-400 group-hover:scale-110 transition-transform" />
            <div>
              <span className="font-bold text-white block">Run Complete Demo</span>
              <p className="text-[10px] text-slate-400">Instantly generate synthetic E2E case</p>
            </div>
          </button>

          <Link
            href="/cases"
            className="p-4 bg-[#0D1426] hover:bg-[#151F35] border border-[#26324A] hover:border-red-500/40 rounded-xl space-y-2 transition-all block group"
          >
            <AlertOctagon className="h-5 w-5 text-red-400 group-hover:scale-110 transition-transform" />
            <div>
              <span className="font-bold text-white block">Test SOS Simulation</span>
              <p className="text-[10px] text-slate-400">Open a case, then use the SOS panel on the case page</p>
            </div>
          </Link>

          <Link
            href="/research"
            className="p-4 bg-[#0D1426] hover:bg-[#151F35] border border-[#26324A] hover:border-emerald-500/40 rounded-xl space-y-2 transition-all block group"
          >
            <BarChart3 className="h-5 w-5 text-emerald-400 group-hover:scale-110 transition-transform" />
            <div>
              <span className="font-bold text-white block">Research Results</span>
              <p className="text-[10px] text-slate-400">View Classical vs VQC benchmark tables</p>
            </div>
          </Link>
        </div>
      </div>

      {/* RECENT CASES TABLE */}
      <div className="dash-card p-4 sm:p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[#26324A] pb-4 gap-2">
          <div className="flex items-center space-x-2 min-w-0">
            <FolderOpen className="h-5 w-5 text-blue-400 shrink-0" />
            <h2 className="text-base font-bold text-white">Recent Research Cases</h2>
          </div>
          <Link href="/create-case" className="text-xs text-blue-400 hover:underline font-semibold shrink-0">
            + New Assessment
          </Link>
        </div>

        {loading ? (
          <div className="text-slate-500 text-xs py-8 text-center">Loading cases from MongoDB...</div>
        ) : cases.length === 0 ? (
          <div className="text-slate-500 text-xs py-8 text-center">No research cases found. Create a new case to begin.</div>
        ) : (
          <>
            {/* Mobile cards */}
            <div className="sm:hidden space-y-3">
              {cases.map((c) => (
                <Link
                  key={c.case_id}
                  href={`/cases/${c.case_id}`}
                  className="block p-4 rounded-xl border border-[var(--border-card)] bg-[var(--bg-card-sub)] space-y-2"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-sm font-semibold text-[var(--text-main)]">
                      {c.case_id.substring(0, 8)}...
                    </span>
                    <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold ${
                      c.status === "analyzed"
                        ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                        : "bg-amber-500/20 text-amber-400 border border-amber-500/30"
                    }`}>
                      {c.status}
                    </span>
                  </div>
                  <p className="text-xs text-[var(--text-muted)]">
                    {new Date(c.created_at).toLocaleString()}
                  </p>
                  <p className="text-xs text-[var(--text-sub)]">
                    {c.image_reference && c.questionnaire && c.sensor_summary
                      ? "FULL MULTIMODAL FUSION"
                      : "REDUCED MODALITY MODE"}
                  </p>
                </Link>
              ))}
            </div>

            {/* Desktop table */}
            <div className="hidden sm:block overflow-x-auto">
            <table className="w-full text-xs text-left text-slate-300">
              <thead className="text-[11px] text-slate-400 font-semibold uppercase border-b border-[#26324A]">
                <tr>
                  <th scope="col" className="py-3 px-4">Case ID</th>
                  <th scope="col" className="py-3 px-4">Created At</th>
                  <th scope="col" className="py-3 px-4">Status</th>
                  <th scope="col" className="py-3 px-4">Modality Configuration</th>
                  <th scope="col" className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1E293B]">
                {cases.map((c) => (
                  <tr key={c.case_id} className="hover:bg-[#151F35] transition-colors">
                    <td className="py-3 px-4 font-mono text-white font-semibold">{c.case_id.substring(0, 8)}...</td>
                    <td className="py-3 px-4 text-slate-400">
                      {new Date(c.created_at).toLocaleString()}
                    </td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold ${
                        c.status === "analyzed" 
                          ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30" 
                          : "bg-amber-500/20 text-amber-400 border border-amber-500/30"
                      }`}>
                        {c.status}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-medium text-slate-300">
                      {c.image_reference && c.questionnaire && c.sensor_summary 
                        ? "FULL MULTIMODAL FUSION" 
                        : "REDUCED MODALITY MODE"}
                    </td>
                    <td className="py-3 px-4 text-right">
                      <Link 
                        href={`/cases/${c.case_id}`}
                        className="px-3 py-1 bg-blue-600/20 hover:bg-blue-600/30 text-blue-300 border border-blue-500/30 rounded-lg font-semibold text-xs transition-all inline-block"
                      >
                        Open Case
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
