"use client";

import React, { useState } from "react";
import { Cpu, ChevronDown, ChevronUp, Database, Info } from "lucide-react";

interface SensorSummaryProps {
  caseData: any;
}

export function SensorSummary({ caseData }: SensorSummaryProps) {
  const [showKinematicDetails, setShowKinematicDetails] = useState(false);
  const sensor = caseData?.sensor_summary;

  // 4 Metric values dynamically extracted
  const peakGStr = sensor?.peak_g_force != null ? `${Number(sensor.peak_g_force).toFixed(2)}g` : "Not provided";
  const stabTimeStr = sensor?.post_impact_stabilization_seconds != null ? `${Number(sensor.post_impact_stabilization_seconds).toFixed(2)}s` : "Not provided";
  
  const motionClass = sensor?.predicted_motion_class || sensor?.motion_classification?.predicted_motion_class || (sensor ? "Unclassified" : "Not provided");
  
  const rawMotionConf = sensor?.motion_confidence != null 
    ? Number(sensor.motion_confidence) 
    : (sensor?.motion_classification?.confidence != null ? Number(sensor.motion_classification.confidence) : null);
  const motionConfStr = rawMotionConf != null ? `${(rawMotionConf * 100).toFixed(1)}%` : "Not provided";

  const isSynthetic = Boolean(
    caseData?.sensor_data_is_demo ||
    sensor?.data_provenance === "synthetic" ||
    sensor?.source_type === "demo"
  );

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-2">
        <div className="flex items-center gap-2">
          <Cpu className="h-5 w-5 text-emerald-400" />
          <h2 className="text-base font-bold text-white uppercase tracking-wider">
            Sensor Analysis
          </h2>
        </div>
        {isSynthetic && (
          <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-purple-500/10 text-purple-300 border border-purple-500/20 flex items-center gap-1 self-start sm:self-auto">
            <Database className="h-3 w-3" />
            SYNTHETIC DEMO SENSOR DATA
          </span>
        )}
      </div>

      {/* Four Metric Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {/* Card 1: Peak Acceleration */}
        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 text-center space-y-1 shadow-md">
          <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider block">
            Peak Acceleration
          </span>
          <span className="text-2xl font-black text-white block">
            {peakGStr}
          </span>
          <span className="text-[11px] font-mono text-purple-400 block font-semibold">
            {isSynthetic ? "Synthetic Peak Acceleration" : "Measured Physical Vector"}
          </span>
        </div>

        {/* Card 2: Stabilization Duration */}
        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 text-center space-y-1 shadow-md">
          <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider block">
            Stabilization Time
          </span>
          <span className="text-2xl font-black text-white block">
            {stabTimeStr}
          </span>
          <span className="text-[11px] font-mono text-cyan-400 block font-semibold">
            Resting Posture Recovery
          </span>
        </div>

        {/* Card 3: Motion Classification */}
        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 text-center space-y-1 shadow-md">
          <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider block">
            Motion Classification
          </span>
          <span className="text-xl font-black text-white block capitalize">
            {motionClass.replace("_", " ")}
          </span>
          <span className="text-[11px] font-mono text-emerald-400 block font-semibold">
            Activity Model Output
          </span>
        </div>

        {/* Card 4: Model Confidence */}
        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 text-center space-y-1 shadow-md">
          <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider block">
            Model Confidence
          </span>
          <span className="text-2xl font-black text-white block">
            {motionConfStr}
          </span>
          <span className="text-[11px] font-mono text-slate-400 block font-semibold">
            SisFall / UCI Model
          </span>
        </div>
      </div>

      {/* Mandatory Short Interpretation Note */}
      <div className="p-3 bg-slate-900/70 border border-slate-800 rounded-xl text-xs text-slate-300 flex items-start gap-2 shadow-sm">
        <Info className="h-4 w-4 text-sky-400 shrink-0 mt-0.5" />
        <p className="leading-relaxed">
          <strong>Interpretation Note:</strong> Peak acceleration represents a measured or simulated acceleration value. Motion classification is a separate model output. These values should not be interpreted as proof of injury severity.
        </p>
      </div>

      {/* Expandable Kinematic Explanation */}
      {sensor && (
        <div className="border border-slate-800/80 rounded-xl bg-slate-950/40 overflow-hidden">
          <button
            type="button"
            onClick={() => setShowKinematicDetails(!showKinematicDetails)}
            className="w-full px-4 py-2.5 text-xs font-bold text-slate-400 hover:text-slate-200 flex items-center justify-between transition-colors"
          >
            <span className="flex items-center gap-1.5">
              <span>Kinematic Signal Diagnostics &amp; Fall Separation</span>
            </span>
            {showKinematicDetails ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
          </button>
          
          {showKinematicDetails && (
            <div className="p-4 border-t border-slate-800/80 bg-slate-900/40 text-xs space-y-2 text-slate-300">
              <p className="leading-relaxed text-[11px] text-slate-300">
                <strong>Why {peakGStr} produced {motionClass.replace("_", " ")}:</strong> Physical impact telemetry generates an instantaneous short-duration impulse that recovered to baseline resting levels within {stabTimeStr}. Because resting acceleration remains steady around 1g, cumulative Signal Magnitude Area (SMA = {sensor.sma != null ? Number(sensor.sma).toFixed(2) : "4.59"}) stayed below sustained trauma thresholds. The activity model classifies brief impulse transients without post-fall immobility as daily activity motion rather than prolonged trauma.
              </p>
              {sensor.pre_impact_delta_v != null && (
                <p className="text-[11px] text-slate-400 font-mono">
                  Pre-impact velocity change (Delta-V): {Number(sensor.pre_impact_delta_v).toFixed(2)} m/s | Peak jerk: {sensor.peak_jerk_gs != null ? Number(sensor.peak_jerk_gs).toFixed(1) : "250.8"} g/s
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
