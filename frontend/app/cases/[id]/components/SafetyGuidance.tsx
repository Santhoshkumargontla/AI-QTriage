"use client";

import React from "react";
import { AlertOctagon, HeartPulse, LifeBuoy } from "lucide-react";

interface SafetyGuidanceProps {
  caseData: any;
  sosStatus: any;
}

export function SafetyGuidance({ caseData, sosStatus }: SafetyGuidanceProps) {
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

  // Distinguish SOS Status accurately
  const activeSos = sosStatus?.status || sosStatus?.sos_status || caseData?.sos_status;
  const isDelivered = activeSos === "delivered" || activeSos === "sent" || activeSos === "twilio_accepted";
  const isCountdown = activeSos === "countdown" || activeSos === "triggered";
  const isLocalSimulation = !isDelivered && (
    activeSos === "TWILIO_NOT_CONFIGURED" ||
    activeSos === "SIMULATION_SAFE" ||
    caseData?.sos_delivery_outcome === "SIMULATION_SAFE" ||
    caseData?.sos_delivery_mode === "SIMULATION" ||
    !activeSos
  );

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
        <span className="text-xs text-slate-400 font-mono">
          Clinical Rules Engine + Gemini Grounding
        </span>
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
              Based on user-reported acute warning indicators, prompt in-person medical evaluation by a licensed healthcare professional is strongly advised.
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
            <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-1.5">
              <LifeBuoy className="h-4 w-4" />
              Immediate Pre-Clinical First-Aid Recommendations
            </span>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1 text-xs">
              {immediateSteps.map((step: string, idx: number) => (
                <div key={idx} className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-850 flex items-start gap-2 text-slate-300">
                  <span className="text-emerald-400 font-bold font-mono shrink-0">{idx + 1}.</span>
                  <span className="leading-snug text-[11px]">{step}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column (1 span): SOS Status & Prototype Guardrails */}
        <div className="space-y-4">
          {/* SOS Dispatch Status Card */}
          <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                Emergency SOS Dispatch
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
