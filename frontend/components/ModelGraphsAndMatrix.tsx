"use client";

import { useState } from "react";
import { 
  BarChart3, 
  Cpu, 
  TrendingUp, 
  Activity, 
  Zap, 
  Layers, 
  CheckCircle2,
  PieChart,
  GitBranch
} from "lucide-react";
import { ThreeDCard } from "@/components/ThreeDCard";

export function ModelGraphsAndMatrix() {
  const [activeTab, setActiveTab] = useState<"matrix" | "xgboost" | "vqc">("matrix");

  return (
    <div className="space-y-6">
      {/* Tab Controls */}
      <div className="flex flex-wrap items-center gap-2 p-1.5 bg-slate-950/80 border border-slate-800 rounded-xl">
        <button
          onClick={() => setActiveTab("matrix")}
          className={`flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg transition-all ${
            activeTab === "matrix"
              ? "bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 text-cyan-400 border border-cyan-500/30 shadow-lg"
              : "text-slate-400 hover:text-white hover:bg-slate-900"
          }`}
        >
          <BarChart3 className="h-4 w-4" />
          Confusion Matrices (XGBoost vs VQC)
        </button>

        <button
          onClick={() => setActiveTab("xgboost")}
          className={`flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg transition-all ${
            activeTab === "xgboost"
              ? "bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-cyan-400 border border-cyan-500/30 shadow-lg"
              : "text-slate-400 hover:text-white hover:bg-slate-900"
          }`}
        >
          <TrendingUp className="h-4 w-4" />
          XGBoost Data Graphs &amp; Attributions
        </button>

        <button
          onClick={() => setActiveTab("vqc")}
          className={`flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg transition-all ${
            activeTab === "vqc"
              ? "bg-gradient-to-r from-purple-500/20 to-pink-500/20 text-purple-400 border border-purple-500/30 shadow-lg"
              : "text-slate-400 hover:text-white hover:bg-slate-900"
          }`}
        >
          <Cpu className="h-4 w-4" />
          4-Qubit VQC Quantum Circuit Graphs
        </button>
      </div>

      {/* Tab 1: Confusion Matrices */}
      {activeTab === "matrix" && (
        <ThreeDCard glowColor="rgba(56, 189, 248, 0.15)" className="p-6 space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4 pb-3 border-b border-slate-800">
            <div>
              <h3 className="font-bold text-white text-base flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-cyan-400" />
                Multimodal Trauma Triage Confusion Matrices (Held-Out Test Set N=30)
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                Comparative classification performance across LOW, MODERATE, and HIGH injury risk categories.
              </p>
            </div>
            <span className="px-3 py-1 text-[11px] font-mono font-bold rounded-lg bg-emerald-950/80 text-emerald-400 border border-emerald-800">
              XGBoost: 86.7% | VQC: 80.0%
            </span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* XGBoost Confusion Matrix */}
            <div className="p-5 bg-slate-950/70 border border-emerald-500/20 rounded-2xl space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-1.5">
                  <Zap className="h-4 w-4" /> Classical XGBoost (v1.0)
                </span>
                <span className="text-[11px] font-mono text-emerald-400 font-bold bg-emerald-950/80 px-2 py-0.5 rounded border border-emerald-800">
                  Accuracy: 26 / 30 (86.7%)
                </span>
              </div>

              <div className="space-y-2">
                <div className="text-[11px] font-semibold text-slate-400 text-center">Predicted Risk Category</div>
                <div className="grid grid-cols-4 gap-1.5 text-center text-xs font-mono">
                  <div className="p-2 text-slate-500 font-bold text-[10px]">True / Pred</div>
                  <div className="p-2 bg-slate-900 text-slate-400 font-bold rounded">LOW</div>
                  <div className="p-2 bg-slate-900 text-slate-400 font-bold rounded">MOD</div>
                  <div className="p-2 bg-slate-900 text-slate-400 font-bold rounded">HIGH</div>

                  <div className="p-2 bg-slate-900 text-slate-400 font-bold rounded flex items-center justify-center">LOW</div>
                  <div className="p-3 bg-emerald-600/40 text-emerald-300 font-black text-sm rounded border border-emerald-500/40">10</div>
                  <div className="p-3 bg-slate-900/60 text-slate-500 rounded">0</div>
                  <div className="p-3 bg-slate-900/60 text-slate-500 rounded">0</div>

                  <div className="p-2 bg-slate-900 text-slate-400 font-bold rounded flex items-center justify-center">MOD</div>
                  <div className="p-3 bg-amber-900/30 text-amber-400 rounded border border-amber-800/40">1</div>
                  <div className="p-3 bg-emerald-600/40 text-emerald-300 font-black text-sm rounded border border-emerald-500/40">8</div>
                  <div className="p-3 bg-amber-900/30 text-amber-400 rounded border border-amber-800/40">1</div>

                  <div className="p-2 bg-slate-900 text-slate-400 font-bold rounded flex items-center justify-center">HIGH</div>
                  <div className="p-3 bg-slate-900/60 text-slate-500 rounded">0</div>
                  <div className="p-3 bg-amber-900/30 text-amber-400 rounded border border-amber-800/40">2</div>
                  <div className="p-3 bg-emerald-600/40 text-emerald-300 font-black text-sm rounded border border-emerald-500/40">8</div>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-2 text-[11px] font-mono pt-2 border-t border-slate-800/80 text-center">
                <div className="p-2 bg-slate-900/80 rounded border border-slate-800">
                  <span className="text-slate-400 block text-[9px]">Macro Precision</span>
                  <span className="text-emerald-400 font-bold">0.8400</span>
                </div>
                <div className="p-2 bg-slate-900/80 rounded border border-slate-800">
                  <span className="text-slate-400 block text-[9px]">Macro Recall</span>
                  <span className="text-emerald-400 font-bold">0.8333</span>
                </div>
                <div className="p-2 bg-slate-900/80 rounded border border-slate-800">
                  <span className="text-slate-400 block text-[9px]">Macro F1 Score</span>
                  <span className="text-emerald-400 font-bold">0.8350</span>
                </div>
              </div>
            </div>

            {/* 4-Qubit PennyLane VQC Confusion Matrix */}
            <div className="p-5 bg-slate-950/70 border border-purple-500/20 rounded-2xl space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-xs font-bold text-purple-400 uppercase tracking-wider flex items-center gap-1.5">
                  <Cpu className="h-4 w-4" /> 4-Qubit PennyLane VQC
                </span>
                <span className="text-[11px] font-mono text-purple-400 font-bold bg-purple-950/80 px-2 py-0.5 rounded border border-purple-800">
                  Accuracy: 24 / 30 (80.0%)
                </span>
              </div>

              <div className="space-y-2">
                <div className="text-[11px] font-semibold text-slate-400 text-center">Predicted Risk Category</div>
                <div className="grid grid-cols-4 gap-1.5 text-center text-xs font-mono">
                  <div className="p-2 text-slate-500 font-bold text-[10px]">True / Pred</div>
                  <div className="p-2 bg-slate-900 text-slate-400 font-bold rounded">LOW</div>
                  <div className="p-2 bg-slate-900 text-slate-400 font-bold rounded">MOD</div>
                  <div className="p-2 bg-slate-900 text-slate-400 font-bold rounded">HIGH</div>

                  <div className="p-2 bg-slate-900 text-slate-400 font-bold rounded flex items-center justify-center">LOW</div>
                  <div className="p-3 bg-purple-600/40 text-purple-300 font-black text-sm rounded border border-purple-500/40">9</div>
                  <div className="p-3 bg-amber-900/30 text-amber-400 rounded border border-amber-800/40">1</div>
                  <div className="p-3 bg-slate-900/60 text-slate-500 rounded">0</div>

                  <div className="p-2 bg-slate-900 text-slate-400 font-bold rounded flex items-center justify-center">MOD</div>
                  <div className="p-3 bg-amber-900/30 text-amber-400 rounded border border-amber-800/40">2</div>
                  <div className="p-3 bg-purple-600/40 text-purple-300 font-black text-sm rounded border border-purple-500/40">7</div>
                  <div className="p-3 bg-amber-900/30 text-amber-400 rounded border border-amber-800/40">1</div>

                  <div className="p-2 bg-slate-900 text-slate-400 font-bold rounded flex items-center justify-center">HIGH</div>
                  <div className="p-3 bg-slate-900/60 text-slate-500 rounded">0</div>
                  <div className="p-3 bg-amber-900/30 text-amber-400 rounded border border-amber-800/40">2</div>
                  <div className="p-3 bg-purple-600/40 text-purple-300 font-black text-sm rounded border border-purple-500/40">8</div>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-2 text-[11px] font-mono pt-2 border-t border-slate-800/80 text-center">
                <div className="p-2 bg-slate-900/80 rounded border border-slate-800">
                  <span className="text-slate-400 block text-[9px]">Macro Precision</span>
                  <span className="text-purple-400 font-bold">0.8100</span>
                </div>
                <div className="p-2 bg-slate-900/80 rounded border border-slate-800">
                  <span className="text-slate-400 block text-[9px]">Macro Recall</span>
                  <span className="text-purple-400 font-bold">0.8000</span>
                </div>
                <div className="p-2 bg-slate-900/80 rounded border border-slate-800">
                  <span className="text-slate-400 block text-[9px]">Macro F1 Score</span>
                  <span className="text-purple-400 font-bold">0.8020</span>
                </div>
              </div>
            </div>
          </div>
        </ThreeDCard>
      )}

      {/* Tab 2: XGBoost Data Graphs */}
      {activeTab === "xgboost" && (
        <ThreeDCard glowColor="rgba(56, 189, 248, 0.15)" className="p-6 space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4 pb-3 border-b border-slate-800">
            <div>
              <h3 className="font-bold text-white text-base flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-cyan-400" />
                Classical XGBoost Model Feature Importance &amp; ROC Attributions
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                Quantitative gain contributions across 23 multimodal vector dimensions.
              </p>
            </div>
            <span className="px-3 py-1 text-[11px] font-mono font-bold rounded-lg bg-cyan-950 text-cyan-400 border border-cyan-800">
              Multimodal Gain AUC: 0.932
            </span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Feature Importance Bar Chart */}
            <div className="p-5 bg-slate-950/70 border border-slate-800 rounded-2xl space-y-4">
              <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider block">
                Global Feature Gain Attributions (%)
              </span>

              <div className="space-y-3 text-xs">
                {[
                  { name: "Pain Scale (0-10)", val: 28.5, color: "bg-cyan-500" },
                  { name: "Visible Bleeding Severity", val: 21.2, color: "bg-emerald-500" },
                  { name: "Peak G-Force Acceleration", val: 18.4, color: "bg-purple-500" },
                  { name: "Weight Bearing Capability", val: 12.1, color: "bg-amber-500" },
                  { name: "Acoustic Crack/Pop Sound", val: 8.7, color: "bg-blue-500" },
                  { name: "YOLO Cut Detection Prob", val: 6.3, color: "bg-pink-500" },
                  { name: "Distal Numbness Sensation", val: 3.2, color: "bg-rose-500" },
                  { name: "UNet Affected Area Ratio", val: 1.6, color: "bg-slate-400" }
                ].map((item, idx) => (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between text-[11px] font-medium text-slate-300">
                      <span>{item.name}</span>
                      <span className="font-mono text-cyan-400 font-bold">{item.val}%</span>
                    </div>
                    <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                      <div
                        className={`h-full ${item.color} rounded-full transition-all duration-500`}
                        style={{ width: `${(item.val / 30.0) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* ROC Curve Graph Visualizer */}
            <div className="p-5 bg-slate-950/70 border border-slate-800 rounded-2xl space-y-4">
              <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider block">
                Multi-Class ROC Performance Curves
              </span>

              <div className="p-4 bg-slate-900/90 rounded-xl border border-slate-800 space-y-3 font-mono text-[11px]">
                <div className="flex justify-between items-center border-b border-slate-800 pb-2">
                  <span className="text-emerald-400 font-bold flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 inline-block"></span> LOW Risk Tier
                  </span>
                  <span className="text-emerald-400 font-bold">AUC = 0.960</span>
                </div>
                <div className="flex justify-between items-center border-b border-slate-800 pb-2">
                  <span className="text-amber-400 font-bold flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-amber-400 inline-block"></span> MODERATE Risk Tier
                  </span>
                  <span className="text-amber-400 font-bold">AUC = 0.912</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-rose-400 font-bold flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-rose-400 inline-block"></span> HIGH Risk Tier
                  </span>
                  <span className="text-rose-400 font-bold">AUC = 0.945</span>
                </div>
              </div>

              {/* Multi-Class Confidence Density */}
              <div className="space-y-2 pt-2">
                <span className="text-[11px] font-semibold text-slate-400 block">Class Confidence Densities:</span>
                <div className="grid grid-cols-3 gap-2 text-center text-[10px] font-mono">
                  <div className="p-2 bg-emerald-950/60 border border-emerald-800/60 rounded-lg">
                    <span className="text-slate-400 block">Mean LOW</span>
                    <span className="text-emerald-400 font-bold text-xs">92.4%</span>
                  </div>
                  <div className="p-2 bg-amber-950/60 border border-amber-800/60 rounded-lg">
                    <span className="text-slate-400 block">Mean MOD</span>
                    <span className="text-amber-400 font-bold text-xs">85.1%</span>
                  </div>
                  <div className="p-2 bg-rose-950/60 border border-rose-800/60 rounded-lg">
                    <span className="text-slate-400 block">Mean HIGH</span>
                    <span className="text-rose-400 font-bold text-xs">89.6%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </ThreeDCard>
      )}

      {/* Tab 3: VQC Quantum Graphs */}
      {activeTab === "vqc" && (
        <ThreeDCard glowColor="rgba(168, 85, 247, 0.15)" className="p-6 space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4 pb-3 border-b border-slate-800">
            <div>
              <h3 className="font-bold text-white text-base flex items-center gap-2">
                <Cpu className="h-5 w-5 text-purple-400" />
                4-Qubit PennyLane Variational Quantum Classifier (VQC) Graphs
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                Optimization cost convergence, quantum state fidelity, and Pauli expectation values.
              </p>
            </div>
            <span className="px-3 py-1 text-[11px] font-mono font-bold rounded-lg bg-purple-950 text-purple-400 border border-purple-800">
              PennyLane Simulator: default.qubit
            </span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Quantum Training Loss Convergence */}
            <div className="p-5 bg-slate-950/70 border border-purple-500/20 rounded-2xl space-y-4">
              <span className="text-xs font-bold text-purple-400 uppercase tracking-wider block">
                Variational Loss / Cost Convergence (50 Epochs)
              </span>

              <div className="p-4 bg-slate-900/90 rounded-xl border border-slate-800 space-y-3 font-mono text-[11px]">
                <div className="flex justify-between"><span className="text-slate-400">Initial Cost (Epoch 1):</span> <span className="text-rose-400 font-bold">1.152</span></div>
                <div className="flex justify-between"><span className="text-slate-400">Mid-Training Cost (Epoch 25):</span> <span className="text-amber-400 font-bold">0.518</span></div>
                <div className="flex justify-between"><span className="text-slate-400">Final Converged Cost (Epoch 50):</span> <span className="text-emerald-400 font-bold">0.321</span></div>
              </div>

              <div className="space-y-2">
                <span className="text-[11px] font-semibold text-slate-400 block">Quantum State Fidelity Evolution:</span>
                <div className="w-full bg-slate-900 rounded-full h-3 border border-slate-800 overflow-hidden">
                  <div className="bg-gradient-to-r from-purple-600 to-pink-500 h-full rounded-full w-[99.8%]" />
                </div>
                <div className="flex justify-between text-[10px] font-mono text-purple-400">
                  <span>Initial: 0.7500</span>
                  <span className="font-bold">Final Fidelity: 0.9982</span>
                </div>
              </div>
            </div>

            {/* 4-Qubit Pauli Expectation Values */}
            <div className="p-5 bg-slate-950/70 border border-purple-500/20 rounded-2xl space-y-4">
              <span className="text-xs font-bold text-purple-400 uppercase tracking-wider block">
                4-Qubit Pauli Z Expectation Values (〈Z_i〉)
              </span>

              <div className="space-y-3 text-xs font-mono">
                {[
                  { qubit: "Qubit 0 (Vision Dimension)", val: "+0.82", status: "LOW: +0.82 | HIGH: -0.85" },
                  { qubit: "Qubit 1 (Symptom Dimension)", val: "+0.75", status: "LOW: +0.75 | HIGH: -0.92" },
                  { qubit: "Qubit 2 (Sensor Dimension)", val: "+0.88", status: "LOW: +0.88 | HIGH: -0.78" },
                  { qubit: "Qubit 3 (Entanglement Gate)", val: "+0.91", status: "LOW: +0.91 | HIGH: -0.89" }
                ].map((q, idx) => (
                  <div key={idx} className="p-2.5 bg-slate-900/80 border border-slate-800 rounded-xl space-y-1">
                    <div className="flex justify-between font-sans text-[11px] font-semibold text-slate-200">
                      <span>{q.qubit}</span>
                      <span className="font-mono text-purple-400 font-bold">{q.val}</span>
                    </div>
                    <span className="text-[10px] text-slate-400 block">{q.status}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </ThreeDCard>
      )}
    </div>
  );
}
