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
  Info,
  ShieldCheck,
  Sparkles,
  Layers,
  Zap,
  CheckCircle2,
  Award,
  FileCode
} from "lucide-react";
import { ThreeDCard } from "@/components/ThreeDCard";
import { ModelGraphsAndMatrix } from "@/components/ModelGraphsAndMatrix";

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
    <div className="space-y-8 flex flex-col flex-1 pb-12">
      {/* Header Banner */}
      <div className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white flex items-center gap-3">
              <Sparkles className="h-7 w-7 text-cyan-400" />
              Research Mode &amp; Model Benchmarks
            </h1>
            <p className="text-slate-400 text-xs sm:text-sm mt-1">
              Comprehensive performance evaluation, artifact provenance, and QML experimental metrics.
            </p>
          </div>
          <span className="px-3 py-1.5 rounded-full text-xs font-semibold bg-amber-950/80 text-amber-400 border border-amber-700/60 shadow-lg flex items-center gap-1.5">
            <AlertTriangle className="h-3.5 w-3.5" />
            SYNTHETIC HELD-OUT EVALUATION — RESEARCH PROTOTYPE
          </span>
        </div>

        <div className="p-4 bg-slate-900/80 border border-slate-800/80 rounded-2xl space-y-2 text-xs text-slate-300 backdrop-blur-md">
          <p className="font-semibold text-slate-200" suppressHydrationWarning>
            Performance measured on the held-out research test set
            (N={loading ? "…" : isEvaluated ? sampleCount : "unavailable"} samples from canonical evaluations).
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-[11px] font-mono text-slate-400 pt-2 border-t border-slate-800/80">
            <div><strong className="text-slate-300">DATA TYPE:</strong> Synthetic Research Data</div>
            <div><strong className="text-slate-300">REAL PATIENTS:</strong> 0 Paired Records</div>
            <div><strong className="text-slate-300">CLINICAL VALIDATION:</strong> Not Performed</div>
            <div><strong className="text-slate-300">LABEL SOURCE:</strong> Rule-Derived Category</div>
          </div>
        </div>
      </div>

      {/* Warning for limited sample count */}
      {isEvaluated && isLimitedSamples && (
        <div className="p-4 bg-amber-950/30 border border-amber-900/60 rounded-2xl flex items-center space-x-3 text-amber-300 text-xs">
          <AlertTriangle className="h-5 w-5 text-amber-400 flex-shrink-0" />
          <div>
            <span className="font-bold block">Limited evaluation set ({sampleCount} samples)</span>
            <p>Preliminary research evaluation — sample size may be insufficient for real-world clinical claims.</p>
          </div>
        </div>
      )}

      {loading ? (
        <div className="flex flex-col items-center justify-center py-24 space-y-4">
          <Activity className="h-10 w-10 text-cyan-400 animate-spin" />
          <span className="text-slate-400 text-xs font-mono">Loading model registry and research benchmarks...</span>
        </div>
      ) : (
        <div className="space-y-8">
          {/* Quick Metrics Summary Cards (3D Tilt) */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <ThreeDCard glowColor="rgba(56, 189, 248, 0.2)" className="p-5">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Registered Models</span>
                <Layers className="h-5 w-5 text-cyan-400" />
              </div>
              <div className="text-2xl font-black text-white">6 Checkpoints</div>
              <div className="text-[11px] text-emerald-400 font-mono mt-1 flex items-center gap-1">
                <CheckCircle2 className="h-3 w-3" /> 100% SHA-256 Verified
              </div>
            </ThreeDCard>

            <ThreeDCard glowColor="rgba(16, 185, 129, 0.2)" className="p-5">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Vision Models</span>
                <Zap className="h-5 w-5 text-emerald-400" />
              </div>
              <div className="text-2xl font-black text-white">YOLO11 + EffNet</div>
              <div className="text-[11px] text-slate-400 font-mono mt-1">
                96.6% EffNet / 34.5% YOLO mAP
              </div>
            </ThreeDCard>

            <ThreeDCard glowColor="rgba(168, 85, 247, 0.2)" className="p-5">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Quantum Classifier</span>
                <Cpu className="h-5 w-5 text-purple-400" />
              </div>
              <div className="text-2xl font-black text-white">4-Qubit VQC</div>
              <div className="text-[11px] text-purple-400 font-mono mt-1">
                53.3% Acc (PennyLane Sim)
              </div>
            </ThreeDCard>

            <ThreeDCard glowColor="rgba(245, 158, 11, 0.2)" className="p-5">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Multimodal Provenance</span>
                <Database className="h-5 w-5 text-amber-400" />
              </div>
              <div className="text-2xl font-black text-white">0 Paired Patient</div>
              <div className="text-[11px] text-amber-400 font-mono mt-1">
                200 Synthetic Fusion Base
              </div>
            </ThreeDCard>
          </div>

          {/* Twilio Integration Status Card */}
          {twilioConfig && (
            <ThreeDCard glowColor="rgba(16, 185, 129, 0.15)" className="p-6 space-y-4">
              <div className="flex justify-between items-center pb-3 border-b border-slate-800">
                <div className="flex items-center space-x-2">
                  <Activity className="h-5 w-5 text-emerald-400" />
                  <h3 className="font-bold text-white text-base">Twilio Integration Status</h3>
                </div>
                <span className={`text-[10px] font-bold px-2.5 py-1 rounded-lg border uppercase tracking-wider ${getTwilioStatusBadge().style}`}>
                  {getTwilioStatusBadge().label}
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                <div className="p-3.5 bg-slate-950/60 border border-slate-800 rounded-xl space-y-1">
                  <span className="text-[10px] text-slate-400 uppercase font-bold block">Integration Status</span>
                  <span className="font-semibold text-slate-200">{twilioConfig.configured ? "CREDENTIALS CONFIGURED" : "NOT CONFIGURED"}</span>
                </div>

                <div className="p-3.5 bg-slate-950/60 border border-slate-800 rounded-xl space-y-1">
                  <span className="text-[10px] text-slate-400 uppercase font-bold block">Supported Alert Modes</span>
                  <span className="font-semibold text-slate-200">
                    {twilioConfig.configured
                      ? "Local Simulation / Twilio SMS Test"
                      : "Local Simulation only (Twilio SMS not available)"}
                  </span>
                </div>

                <div className="p-3.5 bg-slate-950/60 border border-slate-800 rounded-xl space-y-1">
                  <span className="text-[10px] text-slate-400 uppercase font-bold block">Account SID Reference</span>
                  <span className="font-mono text-slate-300">{twilioConfig.account_sid_suffix ? `SID: ${twilioConfig.account_sid_suffix}` : "Not configured"}</span>
                </div>
              </div>

              <div className="p-3 bg-slate-950/70 border border-slate-800 rounded-xl text-xs text-slate-400 space-y-1">
                <p className="text-slate-300 font-semibold">{twilioConfig.status_message}</p>
                <p className="text-[11px] text-amber-400 font-medium">
                  * Security Note: Authentication tokens are strictly isolated within the backend environment and are never stored in MongoDB or exposed to the frontend.
                </p>
              </div>
            </ThreeDCard>
          )}

          {/* Retrained Model Highlights Grid */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <Award className="h-5 w-5 text-cyan-400" />
                Retrained &amp; Promoted Model Suite Architecture
              </h2>
              <span className="text-xs text-slate-400 font-mono">6 Registered Pipelines</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
              {/* Card 1: YOLO11 */}
              <ThreeDCard glowColor="rgba(56, 189, 248, 0.2)" className="p-5 space-y-3">
                <div className="flex justify-between items-start">
                  <div>
                    <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-cyan-950 text-cyan-400 border border-cyan-800 uppercase">Vision Detector</span>
                    <h3 className="text-base font-bold text-white mt-1">YOLO11 Detection</h3>
                  </div>
                  <span className="text-xs font-bold text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800">PROMOTED</span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Trained on 2,813 images with 3,899 bounding boxes across 6 skin lesion classes using SAHI multi-tile windowing.
                </p>
                <div className="space-y-1.5 text-xs font-mono bg-slate-950/60 p-3 rounded-xl border border-slate-800">
                  <div className="flex justify-between"><span className="text-slate-400">mAP@50:</span> <span className="text-cyan-400 font-bold">34.54%</span></div>
                  <div className="flex justify-between"><span className="text-slate-400">mAP@50-95:</span> <span className="text-slate-200">16.74%</span></div>
                  <div className="flex justify-between"><span className="text-slate-400">Precision / Recall:</span> <span className="text-slate-200">51.59% / 39.42%</span></div>
                </div>
                <div className="text-[10px] font-mono text-slate-500 space-y-1 pt-1 border-t border-slate-800">
                  <div>Path: <code className="text-slate-400">ml/models/vision/yolo11_injury_best.pt</code></div>
                  <div className="truncate">SHA-256: <code className="text-cyan-400">857880192ebf...</code></div>
                </div>
              </ThreeDCard>

              {/* Card 2: EfficientNetV2 */}
              <ThreeDCard glowColor="rgba(16, 185, 129, 0.2)" className="p-5 space-y-3">
                <div className="flex justify-between items-start">
                  <div>
                    <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-emerald-950 text-emerald-400 border border-emerald-800 uppercase">Classifier</span>
                    <h3 className="text-base font-bold text-white mt-1">EfficientNetV2</h3>
                  </div>
                  <span className="text-xs font-bold text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800">PROMOTED</span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Multi-class wound photo classifier supporting 8 classes (6 injury types + Normal + OOD Reject).
                </p>
                <div className="space-y-1.5 text-xs font-mono bg-slate-950/60 p-3 rounded-xl border border-slate-800">
                  <div className="flex justify-between"><span className="text-slate-400">Test Accuracy:</span> <span className="text-emerald-400 font-bold">96.62%</span></div>
                  <div className="flex justify-between"><span className="text-slate-400">Macro F1:</span> <span className="text-slate-200">0.9434</span></div>
                  <div className="flex justify-between"><span className="text-slate-400">Test Samples:</span> <span className="text-slate-200">N=473</span></div>
                </div>
                <div className="text-[10px] font-mono text-slate-500 space-y-1 pt-1 border-t border-slate-800">
                  <div>Path: <code className="text-slate-400">ml/models/vision/efficientnetv2_injury_best.pt</code></div>
                  <div className="truncate">SHA-256: <code className="text-emerald-400">8cb4263e70b1...</code></div>
                </div>
              </ThreeDCard>

              {/* Card 3: ResNet34-UNet */}
              <ThreeDCard glowColor="rgba(59, 130, 246, 0.2)" className="p-5 space-y-3">
                <div className="flex justify-between items-start">
                  <div>
                    <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-blue-950 text-blue-400 border border-blue-800 uppercase">Segmenter</span>
                    <h3 className="text-base font-bold text-white mt-1">ResNet34-UNet</h3>
                  </div>
                  <span className="text-xs font-bold text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800">PROMOTED</span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Subject-aware deduplicated binary wound boundary segmentation network (AZH + wseg + Medetec).
                </p>
                <div className="space-y-1.5 text-xs font-mono bg-slate-950/60 p-3 rounded-xl border border-slate-800">
                  <div className="flex justify-between"><span className="text-slate-400">Mean Dice Score:</span> <span className="text-blue-400 font-bold">0.6418</span></div>
                  <div className="flex justify-between"><span className="text-slate-400">Mean IoU:</span> <span className="text-slate-200">0.5178</span></div>
                  <div className="flex justify-between"><span className="text-slate-400">Precision / Recall:</span> <span className="text-slate-200">0.6930 / 0.7233</span></div>
                </div>
                <div className="text-[10px] font-mono text-slate-500 space-y-1 pt-1 border-t border-slate-800">
                  <div>Path: <code className="text-slate-400">ml/models/vision/unet_injury_best.pt</code></div>
                  <div className="truncate">SHA-256: <code className="text-blue-400">3c7f3f39196d...</code></div>
                </div>
              </ThreeDCard>

              {/* Card 4: Sensor Motion Classifier */}
              <ThreeDCard glowColor="rgba(245, 158, 11, 0.2)" className="p-5 space-y-3">
                <div className="flex justify-between items-start">
                  <div>
                    <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-amber-950 text-amber-400 border border-amber-800 uppercase">50Hz Telemetry</span>
                    <h3 className="text-base font-bold text-white mt-1">Sensor Motion Classifier</h3>
                  </div>
                  <span className="text-xs font-bold text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800">PROMOTED</span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Real SisFall + UCI HAR accelerometer window classifier for impact, fall, and activity detection.
                </p>
                <div className="space-y-1.5 text-xs font-mono bg-slate-950/60 p-3 rounded-xl border border-slate-800">
                  <div className="flex justify-between"><span className="text-slate-400">Test Accuracy:</span> <span className="text-amber-400 font-bold">96.34%</span></div>
                  <div className="flex justify-between"><span className="text-slate-400">Matthews Corr (MCC):</span> <span className="text-slate-200">0.9296</span></div>
                  <div className="flex justify-between"><span className="text-slate-400">Test Windows:</span> <span className="text-slate-200">N=1,228 (4,247 train)</span></div>
                </div>
                <div className="text-[10px] font-mono text-slate-500 space-y-1 pt-1 border-t border-slate-800">
                  <div>Path: <code className="text-slate-400">ml/models/sensor_motion_best.json</code></div>
                  <div className="truncate">SHA-256: <code className="text-amber-400">4980d2150f37...</code></div>
                </div>
              </ThreeDCard>

              {/* Card 5: XGBoost Multimodal */}
              <ThreeDCard glowColor="rgba(56, 189, 248, 0.2)" className="p-5 space-y-3">
                <div className="flex justify-between items-start">
                  <div>
                    <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-cyan-950 text-cyan-400 border border-cyan-800 uppercase">Multimodal Fusion</span>
                    <h3 className="text-base font-bold text-white mt-1">XGBoost Multimodal</h3>
                  </div>
                  <span className="text-xs font-bold text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800">PROMOTED</span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  23-feature synthetic fusion model combining image embeddings, symptom questionnaire, and motion vectors.
                </p>
                <div className="space-y-1.5 text-xs font-mono bg-slate-950/60 p-3 rounded-xl border border-slate-800">
                  <div className="flex justify-between"><span className="text-slate-400">Test Accuracy:</span> <span className="text-cyan-400 font-bold">83.33%</span></div>
                  <div className="flex justify-between"><span className="text-slate-400">Macro F1 / MCC:</span> <span className="text-slate-200">0.7850 / 0.7027</span></div>
                  <div className="flex justify-between"><span className="text-slate-400">Dataset Samples:</span> <span className="text-slate-200">1,000 (200 canonical)</span></div>
                </div>
                <div className="text-[10px] font-mono text-slate-500 space-y-1 pt-1 border-t border-slate-800">
                  <div>Path: <code className="text-slate-400">ml/models/xgboost_best.json</code></div>
                  <div className="truncate">SHA-256: <code className="text-cyan-400">73bb5a5125c3...</code></div>
                </div>
              </ThreeDCard>

              {/* Card 6: Experimental 4-Qubit VQC */}
              <ThreeDCard glowColor="rgba(168, 85, 247, 0.2)" className="p-5 space-y-3">
                <div className="flex justify-between items-start">
                  <div>
                    <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-purple-950 text-purple-400 border border-purple-800 uppercase">QML Experimental</span>
                    <h3 className="text-base font-bold text-white mt-1">4-Qubit VQC</h3>
                  </div>
                  <span className="text-xs font-bold text-purple-400 bg-purple-950/60 px-2 py-0.5 rounded border border-purple-800">ISOLATED</span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  PennyLane Variational Quantum Classifier with PCA 4-component feature reduction and AngleEmbedding.
                </p>
                <div className="space-y-1.5 text-xs font-mono bg-slate-950/60 p-3 rounded-xl border border-slate-800">
                  <div className="flex justify-between"><span className="text-slate-400">Test Accuracy:</span> <span className="text-purple-400 font-bold">53.33%</span></div>
                  <div className="flex justify-between"><span className="text-slate-400">Macro F1 / MCC:</span> <span className="text-slate-200">0.4230 / 0.0960</span></div>
                  <div className="flex justify-between"><span className="text-slate-400">Simulator:</span> <span className="text-slate-200">PennyLane default.qubit</span></div>
                </div>
                <div className="text-[10px] font-mono text-slate-500 space-y-1 pt-1 border-t border-slate-800">
                  <div>Path: <code className="text-slate-400">ml/models/vqc/vqc_weights.npz</code></div>
                  <div className="truncate">SHA-256: <code className="text-purple-400">2db769bec3ab...</code></div>
                </div>
              </ThreeDCard>
            </div>
          </div>

          {/* Model Registry Card — Reproducibility & SHA-256 Hashes */}
          <ThreeDCard glowColor="rgba(56, 189, 248, 0.15)" className="p-6 space-y-4">
            <div className="flex justify-between items-center pb-3 border-b border-slate-800">
              <div className="flex items-center space-x-2">
                <Database className="h-5 w-5 text-cyan-400" />
                <h3 className="font-bold text-white text-base">Model Registry Artifacts (v1.3.0)</h3>
              </div>
              <span className="text-[10px] font-mono px-2.5 py-1 rounded bg-cyan-950 text-cyan-400 border border-cyan-800 flex items-center gap-1">
                <FileCode className="h-3 w-3" /> SCIKIT-LEARN &amp; PYTORCH ALIGNED
              </span>
            </div>

            <div className="overflow-x-auto border border-slate-800 rounded-xl">
              <table className="min-w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-950">
                  <tr className="border-b border-slate-800 text-slate-400 font-semibold">
                    <th className="py-3 px-4">Model Name</th>
                    <th className="py-3 px-4">Status</th>
                    <th className="py-3 px-4">Version</th>
                    <th className="py-3 px-4">Dataset &amp; Provenance</th>
                    <th className="py-3 px-4">Split Counts</th>
                    <th className="py-3 px-4">Held-Out Metric</th>
                    <th className="py-3 px-4">Artifact SHA-256 Hash</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 font-mono text-[11px]">
                  {Object.keys(modelRegistry).length > 0 ? (
                    Object.values(modelRegistry).map((reg: any, idx: number) => (
                      <tr key={idx} className="hover:bg-slate-800/40 transition-colors">
                        <td className="py-3 px-4 font-semibold text-slate-200">
                          <div>{reg.model_name}</div>
                          {yoloClassSupportNote(reg) && (
                            <div className="text-[9px] text-cyan-400 font-sans font-semibold mt-0.5 normal-case">
                              {yoloClassSupportNote(reg)}
                            </div>
                          )}
                          {String(reg.status || "").includes("NOT_TRUSTWORTHY") && (
                            <div className="text-[9px] text-amber-400 font-sans font-semibold mt-0.5 normal-case">
                              Raw OOD may collapse; gates withhold — not clinically reliable
                            </div>
                          )}
                        </td>
                        <td className={`py-3 px-4 font-bold ${
                          String(reg.status || reg.training_status || "").includes("EXPERIMENTAL")
                            ? "text-purple-400"
                            : String(reg.status || reg.training_status || "").includes("NOT_TRUSTWORTHY")
                            ? "text-amber-400"
                            : "text-emerald-400"
                        }`}>
                          {reg.status || reg.training_status || "UNKNOWN"}
                        </td>
                        <td className="py-3 px-4 text-cyan-400">{reg.version}</td>
                        <td className="py-3 px-4 text-slate-400">
                          <div>{reg.training_dataset}</div>
                          {reg.data_provenance_class && (
                            <div className="text-[9px] text-slate-500 font-sans mt-0.5">{reg.data_provenance_class}</div>
                          )}
                        </td>
                        <td className="py-3 px-4" title={reg.display_sample_count_note || ""}>
                          {reg.display_sample_count || reg.sample_count || "N/A"}
                        </td>
                        <td className="py-3 px-4 font-bold text-amber-400">
                          <div>{reg.display_held_out_metric || "N/A"}</div>
                          {reg.metrics?.overall_accuracy != null && reg.metrics?.correct_predictions && (
                            <div className="text-[9px] text-emerald-400 font-sans font-medium mt-0.5 normal-case">
                              {(reg.metrics.overall_accuracy * 100).toFixed(2)}% test accuracy
                            </div>
                          )}
                        </td>
                        <td className="py-3 px-4 text-[10px] text-slate-500 truncate max-w-[160px]" title={reg.artifact_sha256}>
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

            <div className="p-3.5 bg-slate-950/60 border border-slate-800 rounded-xl text-[11px] text-slate-400 space-y-1">
              <span className="font-semibold text-slate-300">Data Provenance Summary:</span>
              <p>
                Split counts indicate train / validation / test partition sizes. Held-out metrics evaluate exclusively on test split samples. Genuinely paired patient multimodal records count: <strong>0</strong>. Synthetic fusion records count: <strong>200 canonical (up to 1,000 supported)</strong>.
              </p>
            </div>
          </ThreeDCard>

          {/* Section: Confusion Matrices, XGBoost Graphs, and VQC Quantum Circuit Graphs */}
          <ModelGraphsAndMatrix />

          {/* Section: Classical vs Quantum metrics comparison */}
          <ThreeDCard glowColor="rgba(168, 85, 247, 0.15)" className="p-6 space-y-6">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <h3 className="font-bold text-base text-white flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-purple-400" />
                Experimental Research Classification Performance (Classical XGBoost vs. 4-Qubit VQC)
              </h3>
              <span className="text-xs text-slate-400 font-mono">Test Sample Count: {isEvaluated ? sampleCount : "N/A"}</span>
            </div>

            {!isEvaluated ? (
              <div className="p-8 text-center bg-slate-950/60 border border-slate-800 rounded-xl space-y-2">
                <Info className="h-8 w-8 text-amber-400 mx-auto" />
                <h4 className="font-bold text-white text-sm">Evaluation Unavailable</h4>
                <p className="text-xs text-slate-400 max-w-md mx-auto">
                  Run <code className="text-cyan-400">python ml/training/train_vqc.py</code> to train the experimental VQC and write held-out comparison metrics.
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Metrics Table */}
                <div className="overflow-x-auto border border-slate-800 rounded-xl">
                  <table className="min-w-full text-left text-xs text-slate-300">
                    <thead className="bg-slate-950">
                      <tr className="border-b border-slate-800 text-slate-400 font-semibold">
                        <th className="py-3 px-4">Metric</th>
                        <th className="py-3 px-4 text-cyan-400">Classical (XGBoost)</th>
                        <th className="py-3 px-4 text-purple-400">Experimental 4-Qubit VQC</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
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
                <div className="bg-slate-950/60 border border-slate-800 p-5 rounded-xl space-y-4 text-xs text-slate-300">
                  <div className="flex items-start gap-2 text-purple-400">
                    <Cpu className="h-4 w-4 mt-0.5 flex-shrink-0" />
                    <span className="font-bold text-white">Selective Classification &amp; Quantum Simulator Notice</span>
                  </div>

                  <div className="p-3.5 bg-slate-900/80 border border-slate-800 rounded-lg space-y-1.5 font-mono text-[11px]">
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
                    <div className="text-[10px] text-amber-400 pt-1.5 border-t border-slate-800">
                      Reason:{" "}
                      {comparison.selective_classification?.reason
                        || "Selective-classification metrics not present in the canonical held-out artifact."}
                    </div>
                  </div>

                  <div className="p-3.5 bg-slate-900/80 border border-slate-800 rounded-lg text-[11px] text-slate-400 space-y-1.5">
                    <p className="font-bold text-slate-200">PennyLane Simulator &amp; Isolation Details:</p>
                    <p className="leading-relaxed">
                      PennyLane <code className="text-purple-300">default.qubit</code> runs on classical CPUs. The 4-Qubit VQC operates on PCA-reduced (4 component) classical feature vectors. The VQC is strictly isolated from clinical triage decisions (<code className="text-purple-300">used_in_main_decision=False</code>).
                    </p>
                  </div>
                </div>
              </div>
            )}
          </ThreeDCard>

          {/* Section: Ablation Studies */}
          {ablation && ablation.ablation_study && ablation.ablation_study.length > 0 && (
            <ThreeDCard glowColor="rgba(16, 185, 129, 0.15)" className="p-6 space-y-5">
              <div className="pb-3 border-b border-slate-800">
                <h3 className="font-bold text-base text-white flex items-center gap-2">
                  <Sliders className="h-5 w-5 text-emerald-400" />
                  Modality Ablation Analysis (Experimental Classification Metric Decay)
                </h3>
              </div>

              <div className="overflow-x-auto border border-slate-800 rounded-xl">
                <table className="min-w-full text-left text-xs text-slate-300">
                  <thead className="bg-slate-950">
                    <tr className="border-b border-slate-800 text-slate-400 font-semibold">
                      <th className="py-3 px-4">Configuration</th>
                      <th className="py-3 px-4">Accuracy</th>
                      <th className="py-3 px-4">MCC</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800 font-mono text-[11px]">
                    {ablation.ablation_study.map((item: any, idx: number) => (
                      <tr key={idx} className="hover:bg-slate-800/40 transition-colors">
                        <td className="py-2.5 px-4 font-semibold text-slate-200 font-sans">{item.configuration}</td>
                        <td className="py-2.5 px-4 font-bold text-emerald-400">{(item.accuracy * 100).toFixed(1)}%</td>
                        <td className="py-2.5 px-4 text-slate-300">{item.mcc.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </ThreeDCard>
          )}

          {/* Mandatory Preliminary Research Disclaimer */}
          <div className="p-5 bg-slate-900/90 border border-slate-800 rounded-2xl text-xs text-slate-400 space-y-2 backdrop-blur-md">
            <span className="font-bold text-amber-400 flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-amber-400" /> PRELIMINARY RESEARCH VALIDATION DISCLAIMER
            </span>
            <p className="leading-relaxed">
              The current experimental evaluation uses synthetic research data and rule-derived labels. Reported metrics demonstrate implementation and experimental pipeline behavior and must not be interpreted as clinical performance, medical diagnosis, or real-world injury classification accuracy.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
