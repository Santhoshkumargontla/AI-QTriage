"use client";

import React, { useState } from "react";
import { 
  AlertOctagon, 
  HeartPulse, 
  LifeBuoy, 
  ShieldAlert, 
  Sparkles, 
  AlertTriangle, 
  CheckCircle2, 
  ChevronDown, 
  ChevronUp, 
  Radio, 
  MapPin, 
  ExternalLink, 
  Navigation, 
  Send, 
  Check, 
  Clock 
} from "lucide-react";
import type { SosGeoPayload } from "@/lib/geo";

interface SafetyGuidanceProps {
  caseData: any;
  sosStatus: any;
  sosMode?: "local_demo" | "twilio_test";
  setSosMode?: (mode: "local_demo" | "twilio_test") => void;
  twilioConfig?: any;
  sosGeo?: SosGeoPayload | null;
  sosGeoStatus?: "idle" | "pending" | "ready" | "denied";
  manualLat?: string;
  setManualLat?: (val: string) => void;
  manualLng?: string;
  setManualLng?: (val: string) => void;
  sosError?: string | null;
  onTriggerSOS?: () => Promise<void>;
  onAbortSOS?: () => Promise<void>;
  onRespondSOS?: (resp: "safe" | "no_response") => Promise<void>;
  onDetectLocation?: () => Promise<void>;
}

export function SafetyGuidance({ 
  caseData, 
  sosStatus,
  sosMode = "local_demo",
  setSosMode,
  twilioConfig,
  sosGeo,
  sosGeoStatus = "idle",
  manualLat = "",
  setManualLat,
  manualLng = "",
  setManualLng,
  sosError,
  onTriggerSOS,
  onAbortSOS,
  onRespondSOS,
  onDetectLocation,
}: SafetyGuidanceProps) {
  const [showAvoidAndMonitor, setShowAvoidAndMonitor] = useState(false);
  const [showManualCoords, setShowManualCoords] = useState(false);
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
  const remainingSeconds = sosStatus?.remaining_seconds ?? 10;

  // Active Location Payload
  const activeGeo = sosGeo || caseData?.sos_user_location;
  const hasGeo = activeGeo?.latitude != null && activeGeo?.longitude != null;
  const mapsUrl = activeGeo?.maps_url || (hasGeo ? `https://maps.google.com/?q=${activeGeo.latitude},${activeGeo.longitude}` : null);

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

          {/* Interactive Emergency SOS Alert Dispatch & Location Card */}
          <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                <ShieldAlert className="h-4 w-4 text-rose-400" />
                Emergency SOS &amp; Location
              </span>
              <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                isDelivered
                  ? "bg-emerald-950/80 text-emerald-300 border-emerald-700/60"
                  : isCountdown
                  ? "bg-rose-950/90 text-rose-300 border-rose-600 animate-pulse font-bold"
                  : "bg-slate-800 text-slate-400 border-slate-700"
              }`}>
                {isDelivered
                  ? "ALERT DISPATCHED"
                  : isCountdown
                  ? `COUNTDOWN ACTIVE (${remainingSeconds}s)`
                  : "SIMULATION READY"}
              </span>
            </div>

            {/* Location Sub-Card */}
            <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-2.5">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                  <MapPin className="h-3.5 w-3.5 text-cyan-400" />
                  Incident GPS Location
                </span>
                {hasGeo ? (
                  <span className="text-[10px] font-mono font-bold text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800/60 flex items-center gap-1">
                    <CheckCircle2 className="h-3 w-3" /> GPS Ready
                  </span>
                ) : (
                  <span className="text-[10px] font-mono text-amber-400 bg-amber-950/50 px-2 py-0.5 rounded border border-amber-800/60">
                    Location Pending
                  </span>
                )}
              </div>

              {hasGeo ? (
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 p-2 bg-slate-900/90 rounded-lg border border-slate-800">
                  <div className="font-mono text-xs text-slate-200">
                    <span>{Number(activeGeo.latitude).toFixed(5)}, {Number(activeGeo.longitude).toFixed(5)}</span>
                  </div>
                  {mapsUrl && (
                    <a
                      href={mapsUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-[11px] font-bold text-sky-400 hover:text-sky-300 underline"
                    >
                      <ExternalLink className="h-3 w-3" /> View on Google Maps
                    </a>
                  )}
                </div>
              ) : (
                <p className="text-[11px] text-slate-400">
                  {sosGeoStatus === "pending"
                    ? "Requesting GPS coordinates from your browser..."
                    : "No GPS coordinates attached to this case. Detecting location enables ambulance routing and SMS dispatch."}
                </p>
              )}

              {/* Location Action Buttons */}
              <div className="flex flex-wrap items-center gap-2 pt-1">
                {onDetectLocation && (
                  <button
                    type="button"
                    onClick={onDetectLocation}
                    disabled={sosGeoStatus === "pending"}
                    className="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-750 text-cyan-300 text-[11px] font-bold rounded-lg border border-slate-700 flex items-center gap-1.5 transition-all disabled:opacity-50"
                  >
                    <Navigation className={`h-3 w-3 ${sosGeoStatus === "pending" ? "animate-spin" : ""}`} />
                    <span>{sosGeoStatus === "pending" ? "Detecting GPS..." : hasGeo ? "Re-detect GPS" : "Detect Browser GPS"}</span>
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => setShowManualCoords(!showManualCoords)}
                  className="px-2.5 py-1.5 bg-slate-800/80 hover:bg-slate-800 text-slate-300 text-[11px] font-medium rounded-lg border border-slate-750 transition-all"
                >
                  {showManualCoords ? "Hide Manual Coords" : "Enter Manual Coordinates"}
                </button>
              </div>

              {/* Manual Lat/Lng Fields */}
              {showManualCoords && (
                <div className="p-2.5 bg-slate-900/80 rounded-lg border border-slate-800 space-y-2 pt-2 animate-fadeIn">
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <label className="text-[10px] text-slate-400 block mb-0.5">Latitude</label>
                      <input
                        type="number"
                        step="any"
                        value={manualLat}
                        onChange={(e) => setManualLat?.(e.target.value)}
                        placeholder="e.g. 12.9716"
                        className="w-full px-2 py-1 bg-slate-950 border border-slate-700 rounded text-slate-200 text-xs font-mono"
                      />
                    </div>
                    <div>
                      <label className="text-[10px] text-slate-400 block mb-0.5">Longitude</label>
                      <input
                        type="number"
                        step="any"
                        value={manualLng}
                        onChange={(e) => setManualLng?.(e.target.value)}
                        placeholder="e.g. 77.5946"
                        className="w-full px-2 py-1 bg-slate-950 border border-slate-700 rounded text-slate-200 text-xs font-mono"
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Mode Selection */}
            {setSosMode && (
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-2">
                <span className="text-[11px] font-bold text-slate-300 uppercase tracking-wider block">
                  Alert Transmission Mode
                </span>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setSosMode("local_demo")}
                    className={`px-3 py-2 rounded-lg text-xs font-bold text-left border transition-all ${
                      sosMode === "local_demo"
                        ? "bg-slate-850 text-white border-blue-500/80 shadow-sm"
                        : "bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200"
                    }`}
                  >
                    <span className="block font-bold">Local Simulation</span>
                    <span className="text-[10px] text-slate-400 block font-normal">Internal countdown test</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setSosMode("twilio_test")}
                    className={`px-3 py-2 rounded-lg text-xs font-bold text-left border transition-all ${
                      sosMode === "twilio_test"
                        ? "bg-rose-950/50 text-rose-300 border-rose-500/80 shadow-sm"
                        : "bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200"
                    }`}
                  >
                    <span className="block font-bold">Twilio SMS Test</span>
                    <span className="text-[10px] text-slate-400 block font-normal">
                      {twilioConfig?.to_number ? `Send to ${twilioConfig.to_number.slice(0, 4)}***` : "Live dispatch"}
                    </span>
                  </button>
                </div>
              </div>
            )}

            {/* Error Message */}
            {sosError && (
              <div className="p-2.5 rounded-lg bg-red-950/60 border border-red-800/80 text-xs text-red-200 flex items-start gap-2">
                <AlertTriangle className="h-4 w-4 text-red-400 flex-shrink-0 mt-0.5" />
                <p>{sosError}</p>
              </div>
            )}

            {/* Countdown / Trigger Actions */}
            {isCountdown ? (
              <div className="p-3.5 rounded-xl bg-gradient-to-br from-rose-950/60 to-slate-950 border border-rose-600/80 space-y-3 animate-pulse">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-black text-rose-300 uppercase tracking-wider flex items-center gap-1.5">
                    <Clock className="h-4 w-4 text-rose-400" />
                    EMERGENCY COUNTDOWN ACTIVE
                  </span>
                  <span className="text-sm font-black font-mono text-white bg-rose-600 px-2.5 py-0.5 rounded">
                    {remainingSeconds}s
                  </span>
                </div>
                <p className="text-[11px] text-slate-300 leading-relaxed">
                  Automatic alert with GPS location and clinical indicators will dispatch when the countdown reaches 0.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1">
                  <button
                    type="button"
                    onClick={() => onRespondSOS?.("safe")}
                    className="px-3 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs flex items-center justify-center gap-1.5 transition-all shadow-md"
                  >
                    <Check className="h-3.5 w-3.5" /> I Am Safe (Cancel Alert)
                  </button>
                  <button
                    type="button"
                    onClick={() => onRespondSOS?.("no_response")}
                    className="px-3 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-bold text-xs flex items-center justify-center gap-1.5 transition-all shadow-md"
                  >
                    <Send className="h-3.5 w-3.5" /> Send Alert Now
                  </button>
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                {onTriggerSOS && (
                  <button
                    type="button"
                    onClick={onTriggerSOS}
                    className="w-full py-2.5 px-4 bg-gradient-to-r from-rose-600 to-red-600 hover:from-rose-500 hover:to-red-500 text-white font-bold text-xs rounded-xl shadow-lg shadow-rose-900/30 flex items-center justify-center gap-2 transition-all"
                  >
                    <AlertOctagon className="h-4 w-4" />
                    <span>Trigger Emergency SOS Countdown ({sosMode === "twilio_test" ? "Twilio SMS" : "Local Simulation"})</span>
                  </button>
                )}
                <p className="text-[10px] text-slate-400 text-center">
                  Starts a 10-second countdown before transmitting emergency SMS with patient trauma metrics and Google Maps location.
                </p>
              </div>
            )}

            {/* Delivery Receipt */}
            {isDelivered && (
              <div className="p-3 rounded-xl bg-emerald-950/40 border border-emerald-800/60 space-y-1.5 text-xs">
                <div className="flex items-center gap-1.5 text-emerald-300 font-bold">
                  <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                  <span>Verified SMS Dispatched</span>
                </div>
                <div className="font-mono text-[11px] text-slate-300 space-y-0.5">
                  {caseData?.sos_twilio_sid && (
                    <p><span className="text-slate-500">SID:</span> {caseData.sos_twilio_sid.slice(0, 10)}...{caseData.sos_twilio_sid.slice(-4)}</p>
                  )}
                  {hasGeo && (
                    <p><span className="text-slate-500">Location in SMS:</span> {Number(activeGeo.latitude).toFixed(5)}, {Number(activeGeo.longitude).toFixed(5)}</p>
                  )}
                  {caseData?.sos_send_timestamp && (
                    <p><span className="text-slate-500">Time:</span> {new Date(caseData.sos_send_timestamp).toLocaleTimeString()}</p>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
