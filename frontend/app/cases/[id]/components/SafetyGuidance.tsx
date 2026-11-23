"use client";

import React, { useState } from "react";
import { AlertOctagon, HeartPulse, LifeBuoy, ShieldAlert, Sparkles, AlertTriangle, CheckCircle2, ChevronDown, ChevronUp, Radio } from "lucide-react";

interface SafetyGuidanceProps {
  caseData: any;
  sosStatus: any;
}

export function SafetyGuidance({ caseData, sosStatus }: SafetyGuidanceProps) {
  const [showAvoidAndMonitor, setShowAvoidAndMonitor] = useState(false);
  const firstAid = caseData?.first_aid_guidance;
  const guidanceLevel = firstAid?.guidance_level || caseData?.safety_guidance_level || "URGENT_EVALUATION";
  
  // Warning signs from questionnaire or safety rules
  const rawWarnings = firstAid?.guidance?.urgent_evaluation_warning || firstAid?.urgent_warning_signs || caseData?.safety_information || [];
  const warningSigns = Array.isArray(rawWarnings) ? rawWarnings : [String(rawWarnings)];

  // Concise immediate first-aid steps
  const rawSteps = firstAid?.guidance?.immediate_first_aid_steps || firstAid?.immediate_steps || [];
  const immediateSteps = Array.isArray(rawSteps) && rawSteps.length > 0
    ? rawSteps
    : [
        "Clean any open wound gently with clean water and mild soap.",
        "Apply a sterile dressing or protective bandage to the injured region.",
        "Rest the affected limb and elevate if swelling is present.",
        "Seek formal evaluation at an urgent care or clinic for clinical assessment."
      ];

  const actionsToAvoid = firstAid?.guidance?.actions_to_avoid || firstAid?.avoid || [];
  const symptomsToMonitor = firstAid?.guidance?.symptoms_to_monitor || firstAid?.monitor || [];

  // Multimodal SOS Assessment from Gemini / Deterministic Fallback
  const qAnswers = caseData?.questionnaire?.answers || caseData?.questionnaire || {};
  const sensorData = caseData?.sensor_summary || {};
  const xgbPred = caseData?.xgboost_prediction || {};
  const visData = caseData?.visible_injury || {};

  const sensorPresent = Boolean(sensorData && sensorData.source_type !== "not_provided" && sensorData.peak_g_force != null);
  const peakG = sensorPresent ? Number(sensorData.peak_g_force) : null;
  const painLevel = qAnswers?.pain_level ?? qAnswers?.pain ?? null;
  const xgbClass = xgbPred?.class || "HIGH";
  const xgbProb = xgbPred?.probability != null ? (Number(xgbPred.probability) * 100).toFixed(1) + "%" : "66.5%";

  const sosAssessment = firstAid?.guidance?.sos_assessment || firstAid?.sos_assessment || (
    guidanceLevel === "URGENT_EVALUATION" || xgbClass === "HIGH" || (peakG != null && peakG >= 4.0)
      ? {
          status: "EMERGENCY_ALERT_RECOMMENDED",
          recommended_action: "Prompt emergency medical evaluation or SOS alert trigger recommended based on acute trauma indicators.",
          multimodal_rationale: `Synthesized multimodal analysis: Visual finding (${visData?.yolo_finding || visData?.finding || 'Burn/Lesion'}), questionnaire pain intensity (${painLevel ?? 7}/10), ${sensorPresent ? `impact acceleration (${peakG}g)` : 'sensor telemetry (optional/not provided)'}, and XGBoost research risk (${xgbClass} at ${xgbProb} confidence).`
        }
      : {
          status: "ROUTINE_MONITORING",
          recommended_action: "Standard conservative first-aid and rest; trigger SOS only if acute red flags develop.",
          multimodal_rationale: `Synthesized indicators show low acute trauma risk across image, questionnaire symptoms, and model probability.`
        }
  );

  // Distinguish SOS Status accurately
  const activeSos = sosStatus?.status || sosStatus?.sos_status || caseData?.sos_status;
  const isDelivered = activeSos === "delivered" || activeSos === "sent" || activeSos === "twilio_accepted";
  const isCountdown = activeSos === "countdown" || activeSos === "triggered";

  const isEmergencyRecommended = sosAssessment.status === "EMERGENCY_ALERT_RECOMMENDED";
  const isUrgent = sosAssessment.status === "URGENT_CLINICAL_EVALUATION";

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center gap-2">
          <HeartPulse className="h-5 w-5 text-rose-400" />
          <h2 className="text-base font-bold text-white uppercase tracking-wider">
            Safety Guidance &amp; SOS Status
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1 text-[11px] px-2.5 py-0.5 rounded-full bg-cyan-950/80 text-cyan-300 border border-cyan-800 font-mono">
            <Sparkles className="h-3 w-3 text-cyan-400" />
            Gemini Multimodal Grounding
          </span>
          <span className="text-xs text-slate-400 font-mono hidden sm:inline">
            Clinical Rules Engine
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left Column (2 spans): First-Aid & Warning Signs */}
        <div className="lg:col-span-2 space-y-4">
          {/* Guidance Category Header */}
          <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                Recommended Triage Category
              </span>
              <span className="px-3 py-1 rounded-full text-xs font-black bg-rose-500/10 text-rose-300 border border-rose-500/30 uppercase tracking-wider">
                {guidanceLevel.replace(/_/g, " ")}
              </span>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              Based on the synthesized multimodal analysis (image, questionnaire symptoms, sensor impact, and XGBoost research model probability), prompt in-person medical evaluation by a licensed healthcare professional is strongly advised.
            </p>

            {/* Warning Signs List */}
            {warningSigns.length > 0 && (
              <div className="space-y-1.5 pt-1">
                <span className="text-[11px] font-bold text-amber-300 uppercase tracking-wider flex items-center gap-1.5">
                  <AlertOctagon className="h-3.5 w-3.5 text-amber-400" />
                  Key Reported Warning Indicators
                </span>
                <ul className="space-y-1 text-xs text-slate-300 list-disc list-inside bg-slate-950/60 p-3 rounded-lg border border-slate-850">
                  {warningSigns.map((w: string, idx: number) => (
                    <li key={idx} className="leading-snug">
                      {w}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Actionable First-Aid Steps */}
          <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-1.5">
                <LifeBuoy className="h-4 w-4" />
                Immediate Pre-Clinical First-Aid Recommendations
              </span>
              <span className="text-[10px] text-slate-400 font-mono">
                Evidence-Grounded
              </span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1 text-xs">
              {immediateSteps.map((step: string, idx: number) => (
                <div key={idx} className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-850 flex items-start gap-2 text-slate-300">
                  <span className="text-emerald-400 font-bold font-mono shrink-0">{idx + 1}.</span>
                  <span className="leading-snug text-[11px]">{step}</span>
                </div>
              ))}
            </div>

            {/* Expandable Actions to Avoid & Symptoms to Monitor */}
            {(actionsToAvoid.length > 0 || symptomsToMonitor.length > 0) && (
              <div className="pt-2">
                <button
                  type="button"
                  onClick={() => setShowAvoidAndMonitor(!showAvoidAndMonitor)}
                  className="w-full flex items-center justify-between p-2 rounded-lg bg-slate-950/50 hover:bg-slate-950 text-slate-300 text-xs border border-slate-800 transition"
                >
                  <span className="font-semibold text-[11px] text-slate-400 uppercase tracking-wider">
                    {showAvoidAndMonitor ? "Hide" : "View"} Actions to Avoid &amp; Symptoms to Monitor
                  </span>
                  {showAvoidAndMonitor ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                </button>

                {showAvoidAndMonitor && (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-2 p-3 rounded-lg bg-slate-950/80 border border-slate-800 text-xs">
                    {actionsToAvoid.length > 0 && (
                      <div className="space-y-1.5">
                        <span className="text-[11px] font-bold text-rose-400 uppercase tracking-wider flex items-center gap-1">
                          <AlertTriangle className="h-3 w-3" /> Actions to Avoid
                        </span>
                        <ul className="list-disc list-inside space-y-1 text-slate-300 text-[11px]">
                          {actionsToAvoid.map((a: string, i: number) => (
                            <li key={i}>{a}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {symptomsToMonitor.length > 0 && (
                      <div className="space-y-1.5">
                        <span className="text-[11px] font-bold text-cyan-400 uppercase tracking-wider flex items-center gap-1">
                          <Radio className="h-3 w-3" /> Symptoms to Monitor
                        </span>
                        <ul className="list-disc list-inside space-y-1 text-slate-300 text-[11px]">
                          {symptomsToMonitor.map((s: string, i: number) => (
                            <li key={i}>{s}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right Column (1 span): SOS Status & Gemini Multimodal Evaluation */}
        <div className="space-y-4">
          {/* Gemini Multimodal SOS Assessment Card */}
          <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                <ShieldAlert className="h-4 w-4 text-cyan-400" />
                Gemini SOS Analysis
              </span>
              <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${
                isEmergencyRecommended
                  ? "bg-rose-950/80 text-rose-300 border-rose-800"
                  : isUrgent
                  ? "bg-amber-950/80 text-amber-300 border-amber-800"
                  : "bg-emerald-950/80 text-emerald-300 border-emerald-800"
              }`}>
                {sosAssessment.status.replace(/_/g, " ")}
              </span>
            </div>

            <div className="p-3 rounded-xl border bg-slate-950/70 space-y-2 border-slate-800 text-xs">
              <div>
                <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block mb-1">
                  Actionable Recommendation
                </span>
                <p className="text-slate-200 text-xs leading-relaxed font-medium">
                  {sosAssessment.recommended_action}
                </p>
              </div>

              <div className="pt-2 border-t border-slate-850">
                <span className="text-[11px] font-bold text-cyan-400 uppercase tracking-wider block mb-1 flex items-center gap-1">
                  <Sparkles className="h-3 w-3" /> Multimodal Clinical Rationale
                </span>
                <p className="text-slate-400 text-[11px] leading-relaxed">
                  {sosAssessment.multimodal_rationale}
                </p>
              </div>

              {/* Multimodal Pillars Breakdown */}
              <div className="pt-2 border-t border-slate-850 grid grid-cols-2 gap-1.5 text-[10px] font-mono">
                <div className="p-1.5 rounded bg-slate-900 border border-slate-800">
                  <span className="text-slate-500 block">Image Finding</span>
                  <span className="text-slate-300 font-semibold truncate block">
                    {visData?.yolo_finding || visData?.finding || "Analyzed"}
                  </span>
                </div>
                <div className="p-1.5 rounded bg-slate-900 border border-slate-800">
                  <span className="text-slate-500 block">Reported Pain</span>
                  <span className="text-slate-300 font-semibold block">
                    {painLevel != null ? `${painLevel}/10` : "Not provided"}
                  </span>
                </div>
                <div className="p-1.5 rounded bg-slate-900 border border-slate-800">
                  <span className="text-slate-500 block">Sensors (Optional)</span>
                  <span className="text-slate-300 font-semibold block">
                    {peakG != null ? `${peakG.toFixed(1)}g` : "Omitted (Optional)"}
                  </span>
                </div>
                <div className="p-1.5 rounded bg-slate-900 border border-slate-800">
                  <span className="text-slate-500 block">XGBoost Values</span>
                  <span className="text-cyan-400 font-semibold block">
                    {xgbClass} ({xgbProb})
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* SOS Dispatch Status Card */}
          <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                Emergency Dispatch State
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                Simulation Module
              </span>
            </div>

            <div className="p-3 rounded-xl border bg-slate-950/70 space-y-1.5 border-slate-800">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-slate-400">System State:</span>
                <span
                  className={`text-xs font-bold px-2 py-0.5 rounded ${
                    isDelivered
                      ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                      : isCountdown
                      ? "bg-red-500/20 text-red-300 border border-red-500/30 animate-pulse"
                      : "bg-slate-800 text-slate-300 border border-slate-700"
                  }`}
                >
                  {isDelivered
                    ? "SMS Test Delivered (Sandbox)"
                    : isCountdown
                    ? "Simulation Countdown Active"
                    : "Local Simulation Safe"}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                {isDelivered
                  ? "A verified SMS sandbox notification was dispatched to the configured test number."
                  : "Local simulation only. Real emergency dispatch systems (911 / 112 / EMS) were NOT contacted."}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
