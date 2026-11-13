"use client";

import React, { useState } from "react";
import { Activity, ChevronDown, ChevronUp, Cpu, HelpCircle, ShieldAlert } from "lucide-react";
import { ShapWaterfall } from "@/components/ShapWaterfall";

interface MultimodalResultsProps {
  caseData: any;
}

export function MultimodalResults({ caseData }: MultimodalResultsProps) {
  const [showFullShap, setShowFullShap] = useState(false);

  const xgb = caseData?.xgboost_prediction;
  const xgbClass = xgb?.class || "Unavailable";
  const rawXgbProb = xgb?.probability != null ? Number(xgb.probability) : null;
  const xgbProbStr = rawXgbProb != null ? `${(rawXgbProb * 100).toFixed(1)}%` : "Unavailable";
  const xgbModel = xgb?.model_version || xgb?.model_id || "v1.2.0";

  const vqc = caseData?.quantum_prediction;
  const vqcClass = vqc?.class || "Unavailable";
  const vqcStatus = vqc?.status || (vqc?.experimental ? "EXPERIMENTAL_ONLY" : "Unavailable");

  // SHAP calculations
  const shaps: any[] = Array.isArray(caseData?.shap_explanations) ? caseData.shap_explanations : [];
  const sortedShaps = [...shaps].sort(
    (a, b) => Math.abs(Number(b.shap_value || 0)) - Math.abs(Number(a.shap_value || 0))
  );
  const top5Shaps = sortedShaps.slice(0, 5);

  // Compute residual tail and sums
  const top5Sum = top5Shaps.reduce((acc, s) => acc + Number(s.shap_value || 0), 0);
  const remainingShaps = sortedShaps.slice(5);
  const residualTail = remainingShaps.reduce((acc, s) => acc + Number(s.shap_value || 0), 0);
  const totalFeatureSum = top5Sum + residualTail;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-emerald-400" />
          <h2 className="text-base font-bold text-white uppercase tracking-wider">
            Multimodal AI Results &amp; Explainability
          </h2>
        </div>
        <span className="text-xs text-slate-400 font-mono">
          Classical XGBoost + Experimental VQC
        </span>
      </div>

      {/* Two Model Comparison Cards (Visually Separated) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Card 1: Primary Classical XGBoost Model */}
        <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-3 shadow-md">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-1.5">
              <Activity className="h-4 w-4" />
              Primary Classical Model (XGBoost)
            </span>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-slate-300 border border-slate-700">
              {xgbModel}
            </span>
          </div>

          <div className="flex items-baseline justify-between pt-1">
            <div>
              <span className="text-3xl font-black text-white">
                {xgbClass}
              </span>
              <span className="block text-xs text-slate-400 mt-0.5">
                Predicted Research Category
              </span>
            </div>
            <div className="text-right">
              <span className="text-2xl font-black text-emerald-400 font-mono">
                {xgbProbStr}
              </span>
              <span className="block text-xs text-slate-400 mt-0.5">
                XGBoost Research Model Probability
              </span>
            </div>
          </div>

          <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800 text-[11px] text-slate-400">
            <p>
              <strong className="text-slate-200">Non-Clinical Notice:</strong> This is an algorithmic research classification trained on synthetic rule labels with zero paired clinical records. It is not a clinical severity determination.
            </p>
          </div>
        </div>

        {/* Card 2: Experimental Quantum Classifier (VQC) */}
        <div className="p-5 rounded-2xl bg-purple-950/20 border border-purple-900/40 space-y-3 shadow-md">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-purple-300 uppercase tracking-wider flex items-center gap-1.5">
              <Cpu className="h-4 w-4 text-purple-400" />
              Experimental Quantum Classifier (VQC)
            </span>
            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/10 text-amber-300 border border-amber-500/30">
              Excluded from Main Decision
            </span>
          </div>

          <div className="flex items-baseline justify-between pt-1">
            <div>
              <span className="text-3xl font-black text-purple-200">
                {vqcClass}
              </span>
              <span className="block text-xs text-slate-400 mt-0.5">
                Experimental Quantum Class
              </span>
            </div>
            <div className="text-right">
              <span className="px-2.5 py-1 rounded text-xs font-mono font-bold bg-purple-900/40 text-purple-300 border border-purple-800/50">
                {vqcStatus}
              </span>
              <span className="block text-xs text-slate-400 mt-0.5">
                4-Qubit Simulation
              </span>
            </div>
          </div>

          <div className="p-2.5 rounded-xl bg-slate-950/60 border border-purple-900/30 text-[11px] text-slate-300">
            <p>
              <strong className="text-purple-300">Sandbox Isolation:</strong> The quantum expectation state is evaluated for computational research benchmarking only. It does not contribute to the classical triage decision or emergency escalation.
            </p>
          </div>
        </div>
      </div>

      {/* Compact SHAP Attribution Section */}
      <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4 shadow-md">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <span>TreeSHAP Feature Attributions</span>
              <span className="text-xs text-slate-400 font-normal">
                (Dominant decision drivers for class {xgbClass})
              </span>
            </h3>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Feature contributions decomposed using TreeExplainer on the classical XGBoost model.
            </p>
          </div>
          <button
            type="button"
            onClick={() => setShowFullShap(!showFullShap)}
            className="text-xs font-bold text-sky-400 hover:text-sky-300 flex items-center gap-1 self-start sm:self-auto transition-colors"
          >
            <span>{showFullShap ? "Hide Detailed Breakdown" : "View Full SHAP Breakdown"}</span>
            {showFullShap ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
          </button>
        </div>

        {/* Top 5 SHAP Drivers Cards */}
        {top5Shaps.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
            {top5Shaps.map((item: any, idx: number) => {
              const val = Number(item.shap_value || 0);
              const isPositive = val >= 0;
              return (
                <div
                  key={idx}
                  className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1 text-center"
                >
                  <span className="text-[11px] font-mono text-slate-300 block font-semibold truncate" title={item.feature}>
                    {item.feature.replace(/_/g, " ")}
                  </span>
                  <span
                    className={`text-lg font-black font-mono block ${
                      isPositive ? "text-rose-400" : "text-sky-400"
                    }`}
                  >
                    {isPositive ? "+" : ""}
                    {val.toFixed(3)}
                  </span>
                  <span className="text-[10px] text-slate-400 block font-medium">
                    {isPositive ? "▲ Escalates toward " + xgbClass : "▼ Dampens margin"}
                  </span>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="p-4 text-center text-xs text-slate-400 font-mono">
            SHAP explanations not populated for this record.
          </div>
        )}

        {/* Non-Causal Attribution Notice */}
        <div className="p-2.5 bg-slate-950/40 rounded-xl border border-slate-850 text-[11px] text-slate-400 flex items-start gap-2">
          <HelpCircle className="h-4 w-4 text-slate-500 shrink-0 mt-0.5" />
          <p>
            SHAP values represent additive TreeExplainer marginal contributions to the booster margin for this instance. They are mathematical attributions, not clinical risk percentages or proven causal factors.
          </p>
        </div>

        {/* Expandable Full 23-Feature Breakdown */}
        {showFullShap && shaps.length > 0 && (
          <div className="pt-3 border-t border-slate-800/80 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between text-xs text-slate-300 gap-2">
              <span className="font-bold text-white">Full 23-Feature SHAP Allocation</span>
              <span className="font-mono text-[11px] text-slate-400">
                Top 5 Sum: {top5Sum >= 0 ? "+" : ""}{top5Sum.toFixed(3)} | Residual Tail (18 feats): {residualTail >= 0 ? "+" : ""}{residualTail.toFixed(3)}
              </span>
            </div>

            {/* Complete SHAP Waterfall visualization component */}
            <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800">
              <ShapWaterfall
                features={shaps.map((s: any) => ({
                  name: s.feature ? s.feature.replace(/_/g, " ") : "Feature",
                  value: s.value != null ? String(s.value) : (Number(s.shap_value) >= 0 ? "+contrib" : "-contrib"),
                  shapValue: Number(s.shap_value || 0),
                  description: s.description || `Contribution to margin for ${s.feature}`,
                }))}
                baseValue={caseData?.shap_base_value != null ? Number(caseData.shap_base_value) : null}
                finalScore={caseData?.shap_final_margin != null ? Number(caseData.shap_final_margin) : null}
                isLogOddsScale={true}
                outcomeLabel={caseData?.shap_base_value != null ? `Target Class ${xgbClass} Margin` : `SHAP Contribution Sum (\u2211 \u03c6\u1d62)`}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
