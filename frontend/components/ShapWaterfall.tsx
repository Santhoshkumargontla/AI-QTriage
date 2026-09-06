"use client";

import React, { useState } from "react";

interface ShapFeature {
  name: string;
  value: string | number;
  shapValue: number;
  description: string;
}

interface ShapWaterfallProps {
  features?: ShapFeature[];
  baseValue?: number;
  finalScore?: number;
}

const DEFAULT_FEATURES: ShapFeature[] = [
  { name: "Peak G-Force", value: "8.4g", shapValue: +0.28, description: "High impact telemetry increased triage severity" },
  { name: "Pain Score", value: "8 / 10", shapValue: +0.22, description: "Patient reported severe localized pain" },
  { name: "Affected Area Ratio", value: "14.2%", shapValue: +0.18, description: "Wound ROI covers significant surface area" },
  { name: "Crack/Pop Reported", value: "Yes", shapValue: +0.15, description: "Fracture risk indicator present" },
  { name: "Stabilization Time", value: "1.2s", shapValue: -0.09, description: "Rapid stabilization reduced prolonged trauma probability" },
  { name: "Optical Lux Drop", value: "Normal", shapValue: -0.04, description: "No sudden occlusion detected" },
];

export const ShapWaterfall: React.FC<ShapWaterfallProps> = ({
  features = DEFAULT_FEATURES,
  baseValue = 0.20,
  finalScore = 0.90,
}) => {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  // Find max absolute value for scaling bars
  const maxAbs = Math.max(...features.map((f) => Math.abs(f.shapValue)), 0.35);

  let runningSum = baseValue;

  return (
    <div className="rounded-2xl border border-slate-700/60 bg-slate-900/90 p-6 shadow-2xl backdrop-blur-xl">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            <span className="inline-block h-3 w-3 rounded-full bg-cyan-400 animate-pulse"></span>
            SHAP Waterfall Explainability
          </h3>
          <p className="text-xs text-slate-400">
            Multimodal feature contribution breakdown for classical XGBoost decision
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs font-medium">
          <div className="flex items-center gap-1.5">
            <div className="h-3 w-3 rounded bg-rose-500/80"></div>
            <span className="text-slate-300">Increases Risk (+SHAP)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-3 w-3 rounded bg-emerald-500/80"></div>
            <span className="text-slate-300">Decreases Risk (-SHAP)</span>
          </div>
        </div>
      </div>

      {/* Base Value Line */}
      <div className="mb-3 flex items-center justify-between rounded-lg bg-slate-800/60 px-3 py-1.5 text-xs text-slate-300 border border-slate-700/40">
        <span>Base Expected Value ($E[f(x)]$):</span>
        <span className="font-mono font-bold text-cyan-300">{(baseValue * 100).toFixed(1)}%</span>
      </div>

      {/* Waterfall Bars */}
      <div className="space-y-3">
        {features.map((feat, idx) => {
          const isPositive = feat.shapValue >= 0;
          const absVal = Math.abs(feat.shapValue);
          const barWidthPercent = Math.min(100, (absVal / maxAbs) * 100);
          const startVal = runningSum;
          runningSum += feat.shapValue;

          return (
            <div
              key={idx}
              onMouseEnter={() => setHoveredIdx(idx)}
              onMouseLeave={() => setHoveredIdx(null)}
              className={`group relative rounded-xl border p-3 transition-all duration-200 ${
                hoveredIdx === idx
                  ? "border-cyan-500/50 bg-slate-800/90 shadow-lg scale-[1.01]"
                  : "border-slate-800 bg-slate-800/40 hover:border-slate-700"
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold text-slate-200">{feat.name}</span>
                  <span className="rounded bg-slate-700/80 px-2 py-0.5 font-mono text-xs text-slate-300">
                    {feat.value}
                  </span>
                </div>
                <span
                  className={`font-mono text-xs font-bold ${
                    isPositive ? "text-rose-400" : "text-emerald-400"
                  }`}
                >
                  {isPositive ? `+${(feat.shapValue * 100).toFixed(1)}%` : `${(feat.shapValue * 100).toFixed(1)}%`}
                </span>
              </div>

              {/* Bar Container */}
              <div className="relative h-3.5 w-full rounded-full bg-slate-950/80 overflow-hidden border border-slate-800">
                <div
                  className={`h-full rounded-full transition-all duration-500 ease-out ${
                    isPositive
                      ? "bg-gradient-to-r from-rose-600 via-rose-500 to-pink-400 shadow-[0_0_10px_rgba(244,63,94,0.4)]"
                      : "bg-gradient-to-r from-emerald-600 via-emerald-500 to-teal-400 shadow-[0_0_10px_rgba(16,185,129,0.4)]"
                  }`}
                  style={{ width: `${barWidthPercent}%` }}
                />
              </div>

              {/* Tooltip on hover */}
              {hoveredIdx === idx && (
                <div className="mt-2 rounded-lg bg-slate-950 p-2.5 text-xs text-slate-300 border border-cyan-500/30 animate-in fade-in slide-in-from-top-1">
                  <p className="text-cyan-300 font-medium">{feat.description}</p>
                  <p className="mt-0.5 text-[11px] text-slate-400 font-mono">
                    Shift: {(startVal * 100).toFixed(1)}% &rarr; {(runningSum * 100).toFixed(1)}%
                  </p>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Final Outcome */}
      <div className="mt-4 flex items-center justify-between rounded-xl bg-gradient-to-r from-slate-800 via-slate-800 to-cyan-950/40 p-3 text-sm font-bold text-slate-100 border border-cyan-500/30">
        <span>Final Multimodal Risk Probability:</span>
        <span className="font-mono text-base text-cyan-300">{(finalScore * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
};
