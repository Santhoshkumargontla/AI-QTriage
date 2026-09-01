"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { 
  BarChart3, 
  Cpu, 
  Database, 
  Sliders, 
  Activity, 
  AlertTriangle,
  Info
} from "lucide-react";

function yoloClassSupportNote(reg: Record<string, any>): string | null {
  const name = String(reg.model_name || "");
  if (!name.includes("YOLO11 Detection") || name.includes("Fracture")) return null;
  const classes: string[] = Array.isArray(reg.classes) ? reg.classes.map(String) : [];
  const version = String(reg.version || "runtime");
  let untrained: string[] = [];
  if (Array.isArray(reg.untrained_classes)) {
    untrained = reg.untrained_classes.map(String);
  } else if (reg.untrained_classes && typeof reg.untrained_classes === "object") {
    untrained = Object.keys(reg.untrained_classes);
  } else {
    untrained = ["fracture", "swelling", "Normal", "OOD_Reject"];
  }
  const supported =
    classes.length > 0
      ? `supported: ${classes.join(", ")} (${version})`
      : `supported classes from model.names (${version})`;
  const missing = untrained.length
    ? ` | not in skin head: ${untrained.join(", ")}`
    : "";
  return `${supported}${missing}`;
}

export default function ResearchMode() {
  const [models, setModels] = useState<any[]>([]);
  const [modelRegistry, setModelRegistry] = useState<Record<string, any>>({});
  const [comparison, setComparison] = useState<any>(null);
  const [ablation, setAblation] = useState<any>(null);
  const [twilioConfig, setTwilioConfig] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadResearchData() {
      try {
        const [modelsList, registryData, compData, ablaData, twCfg] = await Promise.all([
          api.getModels(),
          api.getModelRegistry(),
          api.getComparison(),
          api.getAblation(),
          api.getSOSConfig()
        ]);
        setModels(modelsList);
        setModelRegistry(registryData || {});
        setComparison(compData);
        setAblation(ablaData);
        setTwilioConfig(twCfg);
      } catch (err) {
        console.error("Error loading research mode benchmarks:", err);
      } finally {
        setLoading(false);
      }
    }
    loadResearchData();
  }, []);

  const getTwilioStatusBadge = () => {
    if (!twilioConfig || !twilioConfig.enabled) return { label: "NOT CONFIGURED", style: "bg-amber-500/20 text-amber-400 border-amber-500/30" };
    if (twilioConfig.configured) return { label: "CREDENTIALS CONFIGURED", style: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" };
    return { label: "NOT CONFIGURED", style: "bg-red-500/20 text-red-400 border-red-500/30" };
  };

  const isEvaluated = comparison && comparison.status === "evaluated";
  const sampleCount = comparison?.sample_count || 0;
  const isLimitedSamples = sampleCount < 50;

  return (
    <div className="space-y-8 flex flex-col flex-1">
      <div className="space-y-2">
        <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white flex items-center gap-2 flex-wrap">
          Research Mode &amp; Model Benchmarks
          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-950 text-amber-400 border border-amber-800">
            SYNTHETIC HELD-OUT EVALUATION — NOT CLINICALLY VALIDATED
          </span>
        </h1>
        <div className="p-3 bg-slate-900/70 border border-slate-800 rounded-xl space-y-1.5 text-xs text-slate-300">
          <p className="font-semibold text-slate-200" suppressHydrationWarning>
            Performance measured on the held-out synthetic research test set
            (N={loading ? "…" : isEvaluated ? sampleCount : "unavailable"} samples from live predictions).
          </p>
          <div className="flex flex-wrap gap-4 text-[11px] font-mono text-slate-400 pt-1 border-t border-slate-800">
            <span><strong>DATA TYPE:</strong> Synthetic research data</span>
            <span><strong>REAL PATIENT DATA:</strong> Not used</span>
            <span><strong>CLINICAL VALIDATION:</strong> Not performed</span>
            <span><strong>LABEL TYPE:</strong> Rule-Derived Research Category</span>
          </div>
        </div>

      </div>


      {/* Warning for limited sample count */}
      {isEvaluated && isLimitedSamples && (
        <div className="p-4 bg-amber-950/20 border border-amber-900/60 rounded-2xl flex items-center space-x-3 text-amber-300 text-xs">
          <AlertTriangle className="h-5 w-5 text-amber-400 flex-shrink-0" />
          <div>
            <span className="font-bold block">Limited evaluation set ({sampleCount} samples)</span>
            <p>Preliminary research evaluation — insufficient sample size for reliable performance claims.</p>
          </div>
        </div>
      )}

      {loading ? (
        <div className="flex flex-col items-center justify-center py-24 space-y-4">
          <Activity className="h-10 w-10 text-emerald-400 animate-spin" />
          <span className="text-slate-400 text-xs">Loading research data logs...</span>
        </div>
      ) : (
        <div className="space-y-8">
          {/* Twilio Integration Admin Status Card */}
          {twilioConfig && (
            <div className="dash-card p-6 space-y-4">
              <div className="flex justify-between items-center dash-card-header pb-3">
                <div className="flex items-center space-x-2">
                  <Activity className="h-5 w-5 text-emerald-400" />
                  <h3 className="font-bold text-white text-base">Twilio Integration Status</h3>
                </div>
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded-lg border uppercase tracking-wider ${getTwilioStatusBadge().style}`}>
                  {getTwilioStatusBadge().label}
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                <div className="p-3 bg-[#0D1426] border border-[#26324A] rounded-xl space-y-1">
                  <span className="text-[10px] text-slate-400 uppercase font-bold block">Integration Status</span>
                  <span className="font-semibold text-slate-200">{twilioConfig.configured ? "CREDENTIALS CONFIGURED" : "NOT CONFIGURED"}</span>
                </div>

                <div className="p-3 bg-[#0D1426] border border-[#26324A] rounded-xl space-y-1">
                  <span className="text-[10px] text-slate-400 uppercase font-bold block">Supported Alert Modes</span>
                  <span className="font-semibold text-slate-200">
                    {twilioConfig.configured
                      ? "Local Simulation / Twilio SMS Test"
                      : "Local Simulation only (Twilio SMS not available)"}
                  </span>
                </div>

                <div className="p-3 bg-[#0D1426] border border-[#26324A] rounded-xl space-y-1">
                  <span className="text-[10px] text-slate-400 uppercase font-bold block">Account SID Reference</span>
                  <span className="font-mono text-slate-300">{twilioConfig.account_sid_suffix ? `SID: ${twilioConfig.account_sid_suffix}` : "Not configured"}</span>
                </div>
              </div>

              <div className="p-3 bg-[#0D1426] border border-[#26324A] rounded-xl text-xs text-slate-400 space-y-1">
                <p className="text-slate-300 font-semibold">{twilioConfig.status_message}</p>
                <p className="text-[11px] text-amber-400 font-medium">
                  * Security Note: Authentication tokens are strictly isolated within the backend environment and are never stored in MongoDB or exposed to the frontend.
                </p>
              </div>
            </div>
          )}

          {/* Model Registry Card — Reproducibility & SHA-256 Hashes */}
          <div className="dash-card p-6 space-y-4">
            <div className="flex justify-between items-center dash-card-header pb-3">
              <div className="flex items-center space-x-2">
                <Database className="h-5 w-5 text-emerald-400" />
                <h3 className="font-bold text-white text-base">Model Registry Artifacts (v1.1.0)</h3>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">
                SCIKIT-LEARN 1.9.0 ALIGNED
              </span>
            </div>

            <div className="overflow-x-auto border border-[#26324A] rounded-xl">
              <table className="min-w-full text-left text-xs text-slate-300">
                <thead className="bg-[#0D1426]">
                  <tr className="border-b border-[#26324A] text-slate-400 font-semibold">
                    <th className="py-2.5 px-3">Model Name</th>
                    <th className="py-2.5 px-3">Status</th>
                    <th className="py-2.5 px-3">Version</th>
                    <th className="py-2.5 px-3">Dataset</th>
                    <th className="py-2.5 px-3">Split counts</th>
                    <th className="py-2.5 px-3">Held-Out Accuracy / Metric</th>
                    <th className="py-2.5 px-3">Artifact SHA-256 Hash</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1E293B] font-mono text-[11px]">
                  {Object.keys(modelRegistry).length > 0 ? (
                    Object.values(modelRegistry).map((reg: any, idx: number) => (
                      <tr key={idx} className="hover:bg-[#0F172A]">
                        <td className="py-2.5 px-3 font-semibold text-slate-200">
                          <div>{reg.model_name}</div>
                          {yoloClassSupportNote(reg) && (
                            <div className="text-[9px] text-emerald-500/90 font-sans font-semibold mt-0.5 normal-case">
                              {yoloClassSupportNote(reg)}
                            </div>
                          )}
                          {String(reg.status || "").includes("NOT_TRUSTWORTHY") && (
                            <div className="text-[9px] text-amber-500 font-sans font-semibold mt-0.5 normal-case">
                              Raw OOD may collapse; gates withhold — not clinically reliable
                            </div>
                          )}
                        </td>
                        <td className={`py-2.5 px-3 font-bold ${
                          String(reg.status || reg.training_status || "").includes("NOT_TRUSTWORTHY")
                            ? "text-amber-400"
                            : "text-emerald-400"
                        }`}>
                          {reg.status || reg.training_status || "UNKNOWN"}
                        </td>
                        <td className="py-2.5 px-3 text-emerald-400">{reg.version}</td>
                        <td className="py-2.5 px-3 text-slate-400">
                          <div>{reg.training_dataset}</div>
                          {reg.data_provenance_class && (
                            <div className="text-[9px] text-slate-500 font-sans mt-0.5">{reg.data_provenance_class}</div>
                          )}
                        </td>
                        <td className="py-2.5 px-3" title={reg.display_sample_count_note || ""}>
                          {reg.display_sample_count || reg.sample_count || "N/A"}
                        </td>
                        <td className="py-2.5 px-3 font-bold text-amber-400">
                          {reg.display_held_out_metric || "N/A"}
                        </td>
                        <td className="py-2.5 px-3 text-[10px] text-slate-500 truncate max-w-[160px]" title={reg.artifact_sha256}>
                          {reg.artifact_sha256 ? `${reg.artifact_sha256.substring(0, 12)}...` : "N/A"}
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={7} className="py-4 text-center text-slate-500">No registered model artifacts found.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            <div className="p-3 bg-slate-950/50 border border-slate-800 rounded-xl text-[11px] text-slate-400">
              Split counts are train / validation / test sizes from training metadata.
              Held-out metrics use the test split only. Sample count is never the same as held-out N unless labeled as test.
            </div>
            <div className="p-3 bg-amber-950/20 border border-amber-900/40 rounded-xl text-[11px] text-amber-300 space-y-1">
              <span className="font-bold block">Multimodal Dataset Provenance Warning:</span>
              <p>
                Genuinely paired patient records count: <strong>0</strong>. Synthetic multimodal fusion records count: <strong>200</strong>.
                Multimodal records represent synthetic engineering baseline samples. They are not claimed as clinically validated paired patient data.
              </p>
            </div>
          </div>

          {/* Section 1: Classical vs Quantum metrics comparison */}
          <div className="dash-card p-6 space-y-6">
            <div className="flex items-center justify-between dash-card-header pb-4">
              <h3 className="font-bold text-base text-white flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-blue-400" />
                Experimental Research Classification Performance (Classical XGBoost vs. 4-Qubit VQC)
              </h3>
              <span className="text-xs text-slate-400 font-mono">Test Sample Count: {isEvaluated ? sampleCount : "N/A"}</span>
            </div>

            {!isEvaluated ? (
              <div className="p-8 text-center bg-[#0D1426] border border-[#26324A] rounded-xl space-y-2">
                <Info className="h-8 w-8 text-amber-400 mx-auto" />
                <h4 className="font-bold text-white text-sm">Evaluation Unavailable</h4>
                <p className="text-xs text-slate-400 max-w-md mx-auto">
                  Run <code className="text-emerald-400">python ml/training/train_vqc.py</code> to train the experimental VQC and write held-out comparison metrics. This page does not auto-train.
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Metrics Table */}
                <div className="overflow-x-auto border border-[#26324A] rounded-xl">
                  <table className="min-w-full text-left text-xs text-slate-300">
                    <thead className="bg-[#0D1426]">
                      <tr className="border-b border-[#26324A] text-slate-400 font-semibold">
                        <th className="py-3 px-4">Metric</th>
                        <th className="py-3 px-4 text-blue-400">Classical (XGBoost)</th>
                        <th className="py-3 px-4 text-purple-400">Experimental 4-Qubit VQC</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#1E293B]">
                      <tr>
                        <td className="py-3 px-4 font-medium text-slate-200">Correct Predictions</td>
                        <td className="py-3 px-4 font-mono font-bold text-emerald-400">
                          {comparison.classical_xgb.xgb_correct ?? "unavailable"}
                        </td>
                        <td className="py-3 px-4 font-mono font-bold text-purple-400">
                          {comparison.quantum_vqc.vqc_correct ?? "unavailable"}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-3 px-4 font-medium text-slate-200">Accuracy</td>
                        <td className="py-3 px-4 font-mono font-bold text-emerald-400">
                          {comparison.classical_xgb.accuracy != null
                            ? <>{(comparison.classical_xgb.accuracy * 100).toFixed(2)}% <span className="text-[10px] text-slate-500 font-normal">({comparison.classical_xgb.accuracy.toFixed(6)})</span></>
                            : "unavailable"}
                        </td>
                        <td className="py-3 px-4 font-mono font-bold text-purple-400">
                          {comparison.quantum_vqc.accuracy != null
                            ? <>{(comparison.quantum_vqc.accuracy * 100).toFixed(2)}% <span className="text-[10px] text-slate-500 font-normal">({comparison.quantum_vqc.accuracy.toFixed(6)})</span></>
                            : "unavailable"}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-3 px-4 font-medium text-slate-200">Macro Precision</td>
                        <td className="py-3 px-4 font-mono">{comparison.classical_xgb.precision !== undefined && comparison.classical_xgb.precision !== null ? comparison.classical_xgb.precision.toFixed(6) : "unavailable"}</td>
                        <td className="py-3 px-4 font-mono">{comparison.quantum_vqc.precision !== undefined && comparison.quantum_vqc.precision !== null ? comparison.quantum_vqc.precision.toFixed(6) : "unavailable"}</td>
                      </tr>
                      <tr>
                        <td className="py-3 px-4 font-medium text-slate-200">Macro Recall</td>
                        <td className="py-3 px-4 font-mono">{comparison.classical_xgb.recall !== undefined && comparison.classical_xgb.recall !== null ? comparison.classical_xgb.recall.toFixed(6) : "unavailable"}</td>
                        <td className="py-3 px-4 font-mono">{comparison.quantum_vqc.recall !== undefined && comparison.quantum_vqc.recall !== null ? comparison.quantum_vqc.recall.toFixed(6) : "unavailable"}</td>
                      </tr>
                      <tr>
                        <td className="py-3 px-4 font-medium text-slate-200">Macro F1-Score</td>
                        <td className="py-3 px-4 font-mono">{comparison.classical_xgb.macro_f1 != null ? comparison.classical_xgb.macro_f1.toFixed(6) : "unavailable"}</td>
                        <td className="py-3 px-4 font-mono">{comparison.quantum_vqc.macro_f1 != null ? comparison.quantum_vqc.macro_f1.toFixed(6) : "unavailable"}</td>
                      </tr>
                      <tr>
                        <td className="py-3 px-4 font-medium text-slate-200">Matthews Correlation Coefficient (MCC)</td>
                        <td className="py-3 px-4 font-mono">{comparison.classical_xgb.mcc != null ? comparison.classical_xgb.mcc.toFixed(6) : "unavailable"}</td>
                        <td className="py-3 px-4 font-mono">{comparison.quantum_vqc.mcc != null ? comparison.quantum_vqc.mcc.toFixed(6) : "unavailable"}</td>
                      </tr>
                      <tr>
                        <td className="py-3 px-4 font-medium text-slate-200">Expected Calibration Error (ECE)</td>
                        <td className="py-3 px-4 font-mono">{comparison.classical_xgb.ece !== undefined && comparison.classical_xgb.ece !== null ? comparison.classical_xgb.ece.toFixed(6) : "unavailable"}</td>
                        <td className="py-3 px-4 font-mono">{comparison.quantum_vqc.ece !== undefined && comparison.quantum_vqc.ece !== null ? comparison.quantum_vqc.ece.toFixed(6) : "unavailable"}</td>
                      </tr>
                      <tr>
                        <td className="py-3 px-4 font-medium text-slate-200">Brier Score (Multi-class)</td>
                        <td className="py-3 px-4 font-mono">{comparison.classical_xgb.brier_score !== undefined && comparison.classical_xgb.brier_score !== null ? comparison.classical_xgb.brier_score.toFixed(6) : "unavailable"}</td>
                        <td className="py-3 px-4 font-mono">{comparison.quantum_vqc.brier_score !== undefined && comparison.quantum_vqc.brier_score !== null ? comparison.quantum_vqc.brier_score.toFixed(6) : "unavailable"}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>


                {/* Selective Classification & Quantum Simulator Notice Card */}
                <div className="bg-[#0D1426] border border-[#26324A] p-5 rounded-xl space-y-4 text-xs text-slate-300">
                  <div className="flex items-start gap-2 text-purple-400">
                    <Cpu className="h-4 w-4 mt-0.5 flex-shrink-0" />
                    <span className="font-bold text-white">Selective Classification &amp; Quantum Simulator Notice</span>
                  </div>

                  <div className="p-3 bg-[#111A2E] border border-[#26324A] rounded-lg space-y-1.5 font-mono text-[11px]">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Selective Coverage:</span>
                      <span className="text-emerald-400 font-bold">
                        {comparison.selective_classification?.coverage != null
                          ? comparison.selective_classification.coverage
                          : "unavailable"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Accuracy at Coverage:</span>
                      <span className="text-slate-200 font-bold">
                        {comparison.selective_classification?.accuracy_at_coverage != null
                          ? comparison.selective_classification.accuracy_at_coverage
                          : "unavailable"}
                      </span>
                    </div>
                    <div className="text-[10px] text-amber-400 pt-1 border-t border-[#26324A]">
                      Reason:{" "}
                      {comparison.selective_classification?.reason
                        || "Selective-classification metrics not present in the canonical held-out artifact."}
                    </div>
                  </div>

                  <div className="p-3 bg-[#111A2E] border border-[#26324A] rounded-lg text-[11px] text-slate-400 space-y-1">
                    <p className="font-bold text-slate-200">PennyLane Simulator Disclaimer:</p>
                    <p className="leading-relaxed">
                      PennyLane default.qubit is a classical simulation. No quantum advantage or computational superiority is claimed.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Section 2: Ablation Studies */}
          {ablation && ablation.ablation_study && ablation.ablation_study.length > 0 && (
            <div className="dash-card p-6 space-y-5">
              <div className="dash-card-header pb-3">
                <h3 className="font-bold text-base text-white flex items-center gap-2">
                  <Sliders className="h-5 w-5 text-emerald-400" />
                  Modality Ablation Analysis (Experimental Classification Metric Decay)
                </h3>
              </div>

              <div className="overflow-x-auto border border-[#26324A] rounded-xl">
                <table className="min-w-full text-left text-xs text-slate-300">
                  <thead className="bg-[#0D1426]">
                    <tr className="border-b border-[#26324A] text-slate-400 font-semibold">
                      <th className="py-2.5 px-4">Configuration</th>
                      <th className="py-2.5 px-4">Accuracy</th>
                      <th className="py-2.5 px-4">MCC</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1E293B]">
                    {ablation.ablation_study.map((item: any, idx: number) => (
                      <tr key={idx} className="hover:bg-[#151F35] transition-colors">
                        <td className="py-2.5 px-4 font-semibold text-slate-200">{item.configuration}</td>
                        <td className="py-2.5 px-4 font-mono">{(item.accuracy * 100).toFixed(1)}%</td>
                        <td className="py-2.5 px-4 font-mono">{item.mcc.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Model Registry */}
          <div className="dash-card p-6 space-y-4">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <Database className="h-4 w-4 text-blue-400" />
              Model Registry &amp; Training Artifact Status
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
              {models.map((m, idx) => (
                <div key={idx} className="p-4 bg-[#0D1426] border border-[#26324A] rounded-xl space-y-2">
                  <div className="flex justify-between items-start">
                    <span className="font-bold text-sm text-white">{m.model_name}</span>
                    <span className="text-[10px] bg-[#111A2E] text-slate-300 px-2 py-0.5 rounded font-mono border border-[#26324A]">
                      {m.model_version || m.version || "n/a"}
                    </span>
                  </div>
                  <p className={`text-xs font-semibold ${
                    String(m.status || "").includes("NOT_TRUSTWORTHY") || String(m.status || "").includes("UNAVAILABLE")
                      ? "text-amber-400"
                      : m.weights_loaded ? "text-emerald-400" : "text-amber-400"
                  }`}>
                    {m.status}
                  </p>
                  {m.artifact_sha256 && (
                    <p className="text-[10px] font-mono text-slate-500 truncate" title={m.artifact_sha256}>
                      SHA: {m.artifact_sha256.substring(0, 12)}…
                    </p>
                  )}
                  {m.dataset_provenance && (
                    <p className="text-[10px] font-mono text-slate-400 tracking-wider">
                      Provenance: {m.dataset_provenance}
                    </p>
                  )}
                  {m.promotion_status && (
                    <p className="text-[10px] font-mono text-slate-500">
                      Promotion: {m.promotion_status}
                    </p>
                  )}
                  {(m.unsupported_classes || []).length > 0 && (
                    <p className="text-[10px] text-amber-400 font-semibold">
                      Unsupported (in model.names only): {(m.unsupported_classes || []).join(", ")}
                    </p>
                  )}
                  {(m.validated_classes || []).length > 0 && (
                    <p className="text-[10px] text-slate-400">
                      Limited demo classes: {(m.validated_classes || []).join(", ")} (not clinical)
                    </p>
                  )}
                  {m.wound_note && (
                    <p className="text-[10px] text-amber-500/90 leading-snug">{m.wound_note}</p>
                  )}
                  {m.data_provenance && (
                    <p className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
                      Training data: {m.data_provenance}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Mandatory Preliminary Research Disclaimer */}
          <div className="p-4 bg-slate-900/90 border border-slate-800 rounded-2xl text-xs text-slate-400 space-y-2">
            <span className="font-bold text-amber-400 block">PRELIMINARY RESEARCH VALIDATION DISCLAIMER</span>
            <p>
              The current experimental evaluation uses synthetic research data and rule-derived labels. Reported metrics demonstrate implementation and experimental pipeline behavior and must not be interpreted as clinical performance, medical diagnosis, or real-world injury classification accuracy.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

