"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { api, getApiUrl, getUploadUrl } from "@/lib/api";
import { captureBrowserLocation, type SosGeoPayload } from "@/lib/geo";
import { 
  ShieldAlert, 
  Activity, 
  AlertTriangle, 
  FileText, 
  Cpu, 
  Camera, 
  Clock,
  ChevronDown,
  ChevronUp,
  AlertOctagon,
  LifeBuoy,
  HeartPulse,
  Download,
  Sliders,
  CheckCircle2
} from "lucide-react";

export default function CaseDetails() {
  const params = useParams();
  const router = useRouter();
  const caseId = params.id as string;

  const [caseData, setCaseData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  /** Live YOLO model.names — preferred over case snapshot so UI matches promoted checkpoint. */
  const [liveYoloClasses, setLiveYoloClasses] = useState<string[]>([]);
  
  // Overview shows every section. Other tabs jump to that section only.
  const [activeTab, setActiveTab] = useState<"overview" | "image" | "questionnaire" | "sensor" | "ai" | "explainability" | "sos" | "report">("overview");
  const isTab = (...tabs: Array<typeof activeTab>) => tabs.includes(activeTab);
  const selectTab = (tab: typeof activeTab) => {
    setActiveTab(tab);
    if (typeof window !== "undefined") {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  // Accordion details
  const [showTechnical, setShowTechnical] = useState(false);

  
  // SOS countdown status state
  const [sosStatus, setSosStatus] = useState<any>(null);
  const [sosMode, setSosMode] = useState<"local_demo" | "twilio_test">("local_demo");
  const [twilioConfig, setTwilioConfig] = useState<any>(null);
  const [sosGeo, setSosGeo] = useState<SosGeoPayload | null>(null);
  const [sosGeoStatus, setSosGeoStatus] = useState<"idle" | "pending" | "ready" | "denied">("idle");
  const [manualLat, setManualLat] = useState("");
  const [manualLng, setManualLng] = useState("");
  const [sosError, setSosError] = useState<string | null>(null);

  // Vision heatmap vs segmentation mask vs original image view tab
  const [imageTab, setImageTab] = useState<"heatmap" | "mask" | "original">("heatmap");

  useEffect(() => {
    async function loadCase() {
      if (!caseId) return;
      try {
        const [data, twCfg, models] = await Promise.all([
          api.getCase(caseId),
          api.getSOSConfig(),
          api.getModels().catch(() => []),
        ]);
        setCaseData(data);
        setTwilioConfig(twCfg);
        if (data?.sos_user_location?.latitude != null && data?.sos_user_location?.longitude != null) {
          setSosGeo(data.sos_user_location);
          setSosGeoStatus("ready");
        }
        const yoloModel = Array.isArray(models)
          ? models.find((m: any) => String(m.model_name || "").toUpperCase().includes("YOLO"))
          : null;
        const live =
          yoloModel?.classes ||
          yoloModel?.yolo_supported_classes ||
          yoloModel?.supported_classes ||
          [];
        if (Array.isArray(live) && live.length > 0) {
          setLiveYoloClasses(live.map(String));
        }
      } catch (err: any) {
        setError(err.message || "Failed to load research case.");
      } finally {
        setLoading(false);
      }
    }
    loadCase();
  }, [caseId]);


  // SOS Countdown Polling Loop
  useEffect(() => {
    if (!caseId || !caseData) return;
    
    let interval: any = null;
    
    async function checkSOS() {
      try {
        const res = await api.getSOSStatus(caseId);
        if (res) {
          setSosStatus(res);
          if (!["triggered", "countdown", "TWILIO_REQUEST_QUEUED", "twilio_accepted", "sending"].includes(res.status || res.sos_status)) {
            clearInterval(interval);
          }
        }
      } catch (err) {
        console.error("Error polling SOS status:", err);
      }
    }
    
    const currentStatus = sosStatus?.status || sosStatus?.sos_status || caseData?.sos_status;
    if (["triggered", "countdown", "TWILIO_REQUEST_QUEUED", "twilio_accepted", "sending"].includes(currentStatus)) {
      checkSOS();
      interval = setInterval(checkSOS, 1000);
    }
    
    return () => clearInterval(interval);
  }, [caseId, caseData, caseData?.sos_status, sosStatus?.status, sosStatus?.sos_status]);


  const ensureSosGeo = async (): Promise<SosGeoPayload | null> => {
    if (sosGeo?.latitude != null && sosGeo?.longitude != null) return sosGeo;

    const latNum = Number(manualLat);
    const lngNum = Number(manualLng);
    if (
      manualLat.trim() !== "" &&
      manualLng.trim() !== "" &&
      Number.isFinite(latNum) &&
      Number.isFinite(lngNum) &&
      latNum >= -90 &&
      latNum <= 90 &&
      lngNum >= -180 &&
      lngNum <= 180
    ) {
      const maps_url = `https://maps.google.com/?q=${latNum.toFixed(5)},${lngNum.toFixed(5)}`;
      const manual: SosGeoPayload = {
        latitude: latNum,
        longitude: lngNum,
        maps_url,
        location_label: `${latNum.toFixed(5)}, ${lngNum.toFixed(5)}`,
      };
      setSosGeo(manual);
      setSosGeoStatus("ready");
      return manual;
    }

    setSosGeoStatus("pending");
    const geo = await captureBrowserLocation();
    if (geo?.latitude != null && geo?.longitude != null) {
      setSosGeo(geo);
      setSosGeoStatus("ready");
      return geo;
    }
    setSosGeoStatus("denied");
    return null;
  };

  const handleAbortSOS = async () => {
    if (!caseId) return;
    try {
      setSosError(null);
      await api.abortSOS(caseId);
      const res = await api.getSOSStatus(caseId);
      setSosStatus(res);
      const updated = await api.getCase(caseId);
      setCaseData(updated);
    } catch (err) {
      console.error("Failed to abort SOS:", err);
    }
  };

  const handleTriggerSOS = async () => {
    if (!caseId) return;
    try {
      setSosError(null);
      const geo = sosMode === "twilio_test" ? await ensureSosGeo() : sosGeo;
      if (sosMode === "twilio_test" && (geo?.latitude == null || geo?.longitude == null)) {
        setSosError(
          "Twilio SOS requires your GPS location. Allow location access or enter latitude/longitude below."
        );
        return;
      }
      const res = await api.triggerDemoSOS(caseId, sosMode, geo);
      const ev = res?.event || res;
      setSosStatus({
        status: "countdown",
        sos_status: "countdown",
        remaining_seconds: ev?.countdown_seconds || 10,
        message: res?.message || "SOS Countdown Active",
        user_location: res?.user_location || geo,
      });
      if (res?.user_location) setSosGeo(res.user_location);
      const updated = await api.getCase(caseId);
      setCaseData(updated);
    } catch (err: any) {
      console.error("SOS trigger error:", err);
      setSosError(err?.message || "Failed to start SOS countdown.");
    }
  };

  const handleRespondSOS = async (userResp: "safe" | "no_response") => {
    if (!caseId) return;
    try {
      setSosError(null);
      const geo = sosMode === "twilio_test" ? await ensureSosGeo() : sosGeo;
      if (
        sosMode === "twilio_test" &&
        userResp === "no_response" &&
        (geo?.latitude == null || geo?.longitude == null)
      ) {
        setSosError(
          "Twilio SOS requires your GPS location before the SMS can be sent. Allow location access or enter coordinates."
        );
        return;
      }
      const res = await api.respondDemoSOS(caseId, userResp, sosMode, geo);
      setSosStatus({
        status: res.sos_status,
        sos_status: res.sos_status,
        delivery_outcome: res.delivery_outcome || res.sos_status,
        message: res.message,
        twilio_message_sid: res.twilio_message_sid || res.twilio_result?.twilio_message_sid,
        delivery_status: res.provider_status || res.twilio_result?.provider_status || res.twilio_result?.delivery_status,
        provider_status: res.provider_status || res.twilio_result?.provider_status,
        timestamp: res.timestamp,
        failure_reason: res.failure_reason || res.twilio_result?.failure_reason,
        twilio_error_message: res.failure_reason || res.twilio_result?.failure_reason,
        twilio_error_code: res.twilio_result?.error_code,
        user_location: geo || caseData?.sos_user_location,
      });
      const updated = await api.getCase(caseId);
      setCaseData(updated);
    } catch (err: any) {
      console.error("SOS respond error:", err);
      setSosError(err?.message || "Failed to respond to SOS.");
    }
  };

  const storedYoloClasses: string[] = caseData?.visible_injury?.yolo_supported_classes || [];
  const yoloSupportedClasses: string[] =
    liveYoloClasses.length > 0 ? liveYoloClasses : storedYoloClasses;
  const yoloSupportedLower = new Set(yoloSupportedClasses.map((c) => String(c).toLowerCase()));

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] space-y-4">
        <Activity className="h-12 w-12 text-emerald-400 animate-spin" />
        <span className="text-slate-400 text-sm">Retrieving case assessment files...</span>
      </div>
    );
  }

  if (error || !caseData) {
    return (
      <div className="p-6 bg-red-950/20 border border-red-900/60 text-red-200 rounded-xl space-y-4 max-w-xl mx-auto">
        <div className="flex items-center space-x-2">
          <AlertOctagon className="h-5 w-5 text-red-400" />
          <h3 className="font-bold">Error Retrieving Case</h3>
        </div>
        <p>{error || "Case details not found in MongoDB."}</p>
        <button 
          onClick={() => router.push("/")}
          className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-xs"
        >
          Return to Dashboard
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6 sm:space-y-8 flex flex-col flex-1">
      {/* SOS Active Polling Banner */}
      {sosStatus && sosStatus.status === "triggered" && (
        <div className="p-5 bg-red-950/40 border-2 border-red-900 rounded-2xl flex flex-col md:flex-row items-center justify-between gap-4 animate-pulse">
          <div className="flex items-center space-x-3 text-red-200">
            <AlertTriangle className="h-6 w-6 flex-shrink-0 text-red-450" />
            <div className="space-y-1">
              <span className="font-extrabold text-xs block uppercase tracking-wider">SANDBOX EMERGENCY ALERT ACTIVE</span>
              <p className="text-xs text-slate-300">
                {sosStatus.reason || "High impact measured."} Alert dispatches in <strong className="text-red-450 font-bold text-sm font-mono">{sosStatus.remaining_seconds}s</strong>.
              </p>
            </div>
          </div>
          <button
            onClick={handleAbortSOS}
            className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded-lg text-xs font-bold text-white transition-colors"
          >
            I am Safe (Abort Alert)
          </button>
        </div>
      )}

      {/* Demo Mode Alert Banner */}
      {(caseData.is_demo || (caseData.sensor_summary && ["demo", "simulated"].includes(caseData.sensor_summary.source_type))) && (
        <div className="p-4 bg-purple-950/20 border border-purple-850 text-purple-300 rounded-xl flex items-start space-x-3 text-xs">
          <Sliders className="h-5 w-5 flex-shrink-0 mt-0.5 text-purple-400 animate-pulse" />
          <div className="space-y-1">
            <span className="font-bold block text-purple-400 uppercase tracking-wider text-[11px]">DATA PROVENANCE: SYNTHETIC DEMONSTRATION DATA</span>
            <p>
              Research/Demo Data Warning: Results generated using synthetic or simulated data and must not be interpreted as real clinical validation.
            </p>
          </div>
        </div>
      )}

      {/* Disclaimer Banner */}
      <div className="p-4 bg-amber-950/20 border border-amber-900/50 rounded-xl flex items-start space-x-3 text-amber-300 text-xs">
        <ShieldAlert className="h-5 w-5 flex-shrink-0 mt-0.5" />
        <div className="space-y-1">
          <span className="font-bold block">IMPORTANT MEDICAL SAFETY DISCLAIMER</span>
          <p>
            AI-QTriage is an experimental research prototype. It does not diagnose medical conditions or recommend therapy. An ordinary photograph cannot reliably identify fractures or internal injuries. Experimental model categories are not clinical triage decisions.
          </p>
        </div>
      </div>

      {/* Header Info */}
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center border-b border-[var(--border-card)] pb-4 gap-3 sm:gap-4">
        <div className="min-w-0">
          <span className="text-xs sm:text-sm font-semibold text-[var(--text-muted)] font-mono break-all">CASE ID: {caseData.case_id}</span>
          <h1 className="text-xl sm:text-2xl font-bold text-[var(--text-main)] flex flex-col sm:flex-row sm:items-center gap-2 mt-1">
            <span>Multimodal Integration Analysis</span>
            <span className="text-[10px] sm:text-xs font-bold uppercase tracking-wider bg-emerald-950 text-emerald-400 px-2.5 py-0.5 rounded-lg border border-emerald-900/40 w-fit">
              {caseData.sensor_summary 
                ? ((caseData.is_demo || ["demo", "simulated"].includes(caseData.sensor_summary?.source_type)) ? "FULL MULTIMODAL | SYNTHETIC DEMO" : "FULL MULTIMODAL | USER-PROVIDED DATA")
                : "REDUCED MODALITY | SENSOR NOT PROVIDED"}
            </span>
          </h1>
        </div>

        <div className="flex items-start sm:items-center space-x-2 text-xs sm:text-sm text-slate-400 shrink-0">
          <Clock className="h-4 w-4" />
          <span>Completed: {new Date(caseData.created_at).toLocaleString()}</span>
        </div>
      </div>

      {/* Navigation: Overview = full case. Other tabs = that section only. */}
      <div
        className="flex items-center gap-1.5 sm:gap-2 border-b border-[var(--border-card)] pb-3 overflow-x-auto text-xs sm:text-sm font-semibold -mx-1 px-1 scrollbar-thin"
        role="tablist"
        aria-label="Case sections"
      >
        {(["overview", "image", "questionnaire", "sensor", "ai", "explainability", "sos", "report"] as const).map((tab) => (
          <button
            key={tab}
            type="button"
            role="tab"
            aria-selected={activeTab === tab}
            id={`case-tab-${tab}`}
            aria-controls={`case-panel-${tab}`}
            onClick={() => selectTab(tab)}
            className={`px-2.5 sm:px-3.5 py-2 rounded-xl capitalize transition-all whitespace-nowrap shrink-0 ${
              activeTab === tab
                ? "bg-blue-600 text-white font-bold shadow-md shadow-blue-600/20"
                : "text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-[var(--bg-card-sub)]"
            }`}
          >
            {tab === "ai" ? "AI / Quantum" : tab.replace("_", " ")}
          </button>
        ))}
      </div>

      {isTab("overview") && (
        <div id="case-panel-overview" role="tabpanel" aria-labelledby="case-tab-overview" className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
          <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-4 space-y-1">
            <span className="text-[10px] uppercase tracking-wider text-slate-500 font-bold">YOLO11</span>
            <p className="text-sm font-semibold text-white">
              {caseData.visible_injury?.yolo_finding_detected
                ? String(caseData.visible_injury.yolo_finding)
                : "No detection"}
            </p>
            <p className="text-[11px] text-slate-400">
              {caseData.visible_injury?.yolo_confidence != null
                ? `Confidence ${Number(caseData.visible_injury.yolo_confidence).toFixed(2)}`
                : "Keep-threshold 0.25 · not a clinical finding"}
            </p>
          </div>
          <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-4 space-y-1">
            <span className="text-[10px] uppercase tracking-wider text-slate-500 font-bold">EfficientNetV2</span>
            <p className="text-sm font-semibold text-white">
              {caseData.visible_injury?.classifier_finding || "Withheld / unavailable"}
            </p>
            <p className="text-[11px] text-amber-400">
              {caseData.visible_injury?.classifier_model_status || "status unknown"}
            </p>
          </div>
          <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-4 space-y-1">
            <span className="text-[10px] uppercase tracking-wider text-slate-500 font-bold">XGBoost / VQC</span>
            <p className="text-sm font-semibold text-white">
              XGB {caseData.xgboost_prediction?.class || "n/a"}
              <span className="text-slate-500 font-normal"> · </span>
              VQC {caseData.quantum_prediction?.class || "n/a"}
            </p>
            <p className="text-[11px] text-slate-400">
              Fusion labels are synthetic. VQC is experimental only.
            </p>
          </div>
          <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-4 space-y-1">
            <span className="text-[10px] uppercase tracking-wider text-slate-500 font-bold">SOS</span>
            <p className="text-sm font-semibold text-white">
              {sosStatus?.status || sosStatus?.sos_status || caseData.sos_status || "idle"}
            </p>
            <p className="text-[11px] text-slate-400">
              {twilioConfig?.configured ? "Twilio path available" : "LOCAL SOS SIMULATION ONLY"}
            </p>
          </div>
        </div>
      )}

      <div className="space-y-8">
      {/* Emergency SOS Simulation & Twilio Test Panel */}
      {isTab("overview", "sos") && (
      <div id="case-panel-sos" role="tabpanel" aria-labelledby="case-tab-sos" className="bg-slate-900/60 border border-slate-800 p-4 sm:p-6 rounded-2xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-slate-800 pb-3 gap-2">
          <div className="flex items-center space-x-2 min-w-0">
            <AlertOctagon className="h-5 w-5 text-red-400 animate-pulse shrink-0" />
            <h3 className="font-bold text-white text-base truncate">Emergency SOS Simulation</h3>
          </div>
          
          {/* Mode Selector */}
          <div className="flex flex-wrap items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs w-full sm:w-auto">
            <button
              type="button"
              onClick={() => setSosMode("local_demo")}
              className={`flex-1 sm:flex-none px-3 py-2 sm:py-1 rounded-md transition-colors ${
                sosMode === "local_demo"
                  ? "bg-slate-800 text-white font-bold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Local Demo
            </button>
            <button
              type="button"
              onClick={() => {
                setSosMode("twilio_test");
                void ensureSosGeo();
              }}
              className={`flex-1 sm:flex-none px-3 py-2 sm:py-1 rounded-md transition-colors ${
                sosMode === "twilio_test"
                  ? "bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/40"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Twilio Test / Sandbox
            </button>
          </div>
        </div>

        {/* SOS Mode Banner */}
        <div className="text-xs space-y-2">
          {sosMode === "twilio_test" ? (
            <div className="p-3 bg-amber-950/20 border border-amber-900/40 text-amber-200 rounded-xl space-y-2">
              <span className="font-bold block">
                {twilioConfig?.configured ? "Twilio test mode" : "SMS NOT CONFIGURED"}
              </span>
              <p className="text-[11px] leading-relaxed break-words">
                {twilioConfig?.configured
                  ? twilioConfig.status_message
                  : "TWILIO_NOT_CONFIGURED. Countdown stays LOCAL_SIMULATION unless Twilio env vars are set. Canonical: TWILIO_ENABLED, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, TWILIO_TO_NUMBER."}
              </p>
              <div className="pt-1 text-[11px] text-slate-300 space-y-2">
                <p className="font-semibold text-emerald-300">User GPS location (required in SMS)</p>
                {sosGeoStatus === "pending" && <p>Requesting browser GPS… allow location access when prompted.</p>}
                {(sosGeoStatus === "ready" || (sosGeo?.latitude != null && sosGeo?.longitude != null)) && (
                  <p className="break-all">
                    GPS ready: {Number(sosGeo?.latitude).toFixed(5)}, {Number(sosGeo?.longitude).toFixed(5)}
                    {sosGeo?.maps_url ? (
                      <>
                        {" · "}
                        <a href={sosGeo.maps_url} target="_blank" rel="noreferrer" className="underline text-sky-300">
                          Open map
                        </a>
                      </>
                    ) : null}
                  </p>
                )}
                {(sosGeoStatus === "denied" || sosGeoStatus === "idle" || sosGeo?.latitude == null) && (
                  <div className="space-y-2">
                    {sosGeoStatus === "denied" && (
                      <p className="text-amber-300">
                        GPS unavailable or denied. Enter coordinates manually, or retry browser location.
                      </p>
                    )}
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      <label className="space-y-1">
                        <span className="text-slate-400">Latitude</span>
                        <input
                          type="number"
                          step="any"
                          inputMode="decimal"
                          value={manualLat}
                          onChange={(e) => setManualLat(e.target.value)}
                          placeholder="e.g. 12.97160"
                          className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-700 text-slate-100"
                        />
                      </label>
                      <label className="space-y-1">
                        <span className="text-slate-400">Longitude</span>
                        <input
                          type="number"
                          step="any"
                          inputMode="decimal"
                          value={manualLng}
                          onChange={(e) => setManualLng(e.target.value)}
                          placeholder="e.g. 77.59460"
                          className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-700 text-slate-100"
                        />
                      </label>
                    </div>
                    <div className="flex flex-wrap gap-3">
                      <button type="button" className="underline text-sky-300" onClick={() => void ensureSosGeo()}>
                        Use GPS / apply coordinates
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="p-3 bg-slate-950/60 border border-slate-850 text-slate-400 rounded-xl">
              <span><strong>LOCAL SOS SIMULATION ONLY:</strong> Countdown is recorded in MongoDB. SMS is not sent.</span>
            </div>
          )}

          {sosError && (
            <div className="p-3 bg-red-950/30 border border-red-800/50 text-red-200 rounded-xl">
              {sosError}
            </div>
          )}

          {/* Prominent Mandatory Safety Disclaimer */}
          <div className="p-3 bg-amber-950/30 border border-amber-900/50 text-amber-300 rounded-xl font-medium text-[11px] flex items-start sm:items-center gap-2">
            <ShieldAlert className="h-4 w-4 text-amber-400 flex-shrink-0 mt-0.5 sm:mt-0" />
            <span>RESEARCH PROTOTYPE ONLY: NO REAL EMERGENCY SERVICES (911 / 112 / AMBULANCE) ARE EVER CONTACTED.</span>
          </div>
        </div>

        {/* Status display */}
        {(() => {
          const activeSosStatus = sosStatus?.status || sosStatus?.sos_status || caseData?.sos_status;
          const isSosCountdown = ["countdown", "triggered"].includes(activeSosStatus);
          const sosSecondsLeft = Math.max(0, Math.ceil(sosStatus?.remaining_seconds ?? (sosStatus?.countdown_seconds ?? 10)));
          
          // Twilio & Event variables
          const eventId = sosStatus?.event_id || caseData?.active_sos_event_id;
          const msgSid = sosStatus?.twilio_message_sid || caseData?.sos_twilio_sid;
          const maskedSid = msgSid ? `${msgSid.substring(0, 6)}...${msgSid.substring(msgSid.length - 4)}` : null;
          const deliveryStatus = sosStatus?.provider_status || sosStatus?.delivery_status || caseData?.sos_provider_status || caseData?.sos_delivery_status;
          const errDetail = sosStatus?.failure_reason || sosStatus?.twilio_error_message || caseData?.sos_failure_reason || caseData?.sos_twilio_error;
          const errCode = sosStatus?.twilio_error_code || caseData?.sos_twilio_error_code;
          const sendTs = sosStatus?.timestamp || caseData?.sos_send_timestamp;
          const outcome = sosStatus?.delivery_outcome || activeSosStatus;
          
          let statusColor = "bg-slate-600";
          let statusText = activeSosStatus || "NOT TRIGGERED";
          
          if (isSosCountdown) {
            statusColor = "bg-red-500 animate-ping";
            statusText = `COUNTDOWN (${sosSecondsLeft}s)`;
          } else if (activeSosStatus === "cancelled") {
            statusColor = "bg-emerald-400";
            statusText = "CANCELLED (USER SAFE)";
          } else if (activeSosStatus === "sending") {
            statusColor = "bg-yellow-500 animate-pulse";
            statusText = "TWILIO: SENDING REQUEST...";
          } else if (outcome === "TWILIO_REQUEST_QUEUED" || activeSosStatus === "twilio_accepted") {
            statusColor = "bg-blue-400";
            statusText = "TWILIO_REQUEST_QUEUED (SID recorded; not proof of SMS delivery)";
          } else if (outcome === "TWILIO_FAILED" || activeSosStatus === "twilio_failed" || activeSosStatus === "undelivered") {
            statusColor = "bg-red-500";
            statusText = "TWILIO_FAILED";
          } else if (outcome === "TWILIO_NOT_CONFIGURED") {
            statusColor = "bg-amber-400";
            statusText = "TWILIO_NOT_CONFIGURED (no SMS sent)";
          } else if (outcome === "LOCAL_SIMULATION" || activeSosStatus === "demo_triggered") {
            statusColor = "bg-amber-400";
            statusText = "LOCAL_SIMULATION (no SMS sent)";
          } else if (activeSosStatus === "SMS_SENT") {
            statusColor = "bg-red-500";
            statusText = "INVALID STATUS SMS_SENT — treating as unverified";
          }
          
          return (
            <div className="flex flex-col p-4 bg-slate-950/80 rounded-xl border border-slate-850 gap-4 w-full">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="space-y-1">
                  <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wider block">Current SOS State</span>
                  <div className="flex items-center space-x-2">
                    <span className={`h-2.5 w-2.5 rounded-full ${statusColor}`} />
                    <span className="text-sm font-bold text-white uppercase tracking-wide break-words">
                      {statusText}
                    </span>
                  </div>
                  {sosStatus?.message && (
                    <p className="text-xs text-slate-300 italic">{sosStatus.message}</p>
                  )}
                </div>

                <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2 sm:gap-3 w-full sm:w-auto">
                  {isSosCountdown ? (
                    <>
                      <button
                        type="button"
                        onClick={() => handleRespondSOS("safe")}
                        className="px-4 py-2.5 sm:py-2 bg-emerald-500 hover:bg-emerald-450 text-slate-950 font-bold rounded-lg text-xs transition-colors w-full sm:w-auto"
                      >
                        YES — I&apos;m Safe (Cancel SOS)
                      </button>
                      <button
                        type="button"
                        onClick={() => handleRespondSOS("no_response")}
                        className="px-4 py-2.5 sm:py-2 bg-red-600 hover:bg-red-505 text-white font-bold rounded-lg text-xs transition-colors w-full sm:w-auto"
                      >
                        Simulate Countdown Expiry
                      </button>
                    </>
                  ) : (
                    <button
                      type="button"
                      onClick={handleTriggerSOS}
                      className="px-4 py-2.5 sm:py-2 bg-slate-800 hover:bg-slate-750 border border-slate-700 text-slate-200 font-bold rounded-lg text-xs transition-colors flex items-center justify-center space-x-1.5 w-full sm:w-auto"
                    >
                      <AlertOctagon className="h-4 w-4 text-red-400" />
                      <span>Test SOS Countdown ({sosMode === "twilio_test" ? "Twilio Mode" : "Local Mode"})</span>
                    </button>
                  )}
                </div>
              </div>

              {(eventId || msgSid || deliveryStatus || errDetail || sendTs
                || ["TWILIO_REQUEST_QUEUED", "TWILIO_FAILED", "TWILIO_NOT_CONFIGURED", "LOCAL_SIMULATION", "twilio_accepted", "twilio_failed", "demo_triggered"].includes(String(outcome || activeSosStatus))) && (
                <div className="border-t border-slate-850 pt-3 text-xs text-slate-400 space-y-1 bg-slate-900/30 p-3 rounded-lg font-mono">
                  {eventId && (
                    <div>
                      <strong>Event ID:</strong> <span className="text-slate-300">{eventId}</span>
                    </div>
                  )}
                  {msgSid && (
                    <div>
                      <strong>Message SID:</strong> <span className="text-slate-300">{maskedSid}</span>
                    </div>
                  )}
                  {(sosGeo?.latitude != null || caseData?.sos_user_location?.latitude != null) && (
                    <div className="font-sans text-slate-300">
                      <strong>User GPS in SOS:</strong>{" "}
                      {Number(sosGeo?.latitude ?? caseData?.sos_user_location?.latitude).toFixed(5)},{" "}
                      {Number(sosGeo?.longitude ?? caseData?.sos_user_location?.longitude).toFixed(5)}
                      {(sosGeo?.maps_url || caseData?.sos_user_location?.maps_url) && (
                        <>
                          {" · "}
                          <a
                            href={sosGeo?.maps_url || caseData?.sos_user_location?.maps_url}
                            target="_blank"
                            rel="noreferrer"
                            className="underline text-sky-300"
                          >
                            Map link
                          </a>
                        </>
                      )}
                    </div>
                  )}
                  {!msgSid && (outcome === "TWILIO_REQUEST_QUEUED" || outcome === "TWILIO_FAILED" || outcome === "TWILIO_NOT_CONFIGURED") && (
                    <div>
                      <strong>Message SID:</strong> <span className="text-slate-500">none</span>
                    </div>
                  )}
                  {deliveryStatus && (
                    <div>
                      <strong>Provider Status:</strong> <span className="text-slate-300 uppercase">{deliveryStatus}</span>
                    </div>
                  )}
                  {sendTs && (
                    <div>
                      <strong>Timestamp:</strong> <span className="text-slate-300">{sendTs}</span>
                    </div>
                  )}
                  {errDetail && (
                    <div className="text-red-400 mt-1 font-sans">
                      <strong>Failure Reason:</strong> {errDetail}
                      {errCode && <span className="ml-2 font-mono text-[11px] text-red-300">(Error Code: {String(errCode)})</span>}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })()}

      </div>
      )}

      {isTab("overview", "image") && (
      <div className="grid grid-cols-1 gap-8">
        <div className="space-y-8">
          <div id="case-panel-image" role="tabpanel" aria-labelledby="case-tab-image" className="bg-slate-900/50 border border-slate-800 p-6 rounded-2xl space-y-6">
            <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
              <Camera className="h-5 w-5 text-emerald-400" />
              <h3 className="font-bold text-white text-base">Section 1: Visible Image Analysis</h3>
            </div>

            {caseData.visible_injury ? (
              <div className="space-y-4">
                {caseData.visible_injury.source_type === "demo" && (
                  <div className="p-3 bg-purple-950/20 border border-purple-850 text-purple-300 rounded-xl text-xs">
                    <strong>Demo Mode: </strong> Synthetic demonstration image — not a real patient image.
                  </div>
                )}
                {caseData.visible_injury.source_type === "uploaded" && caseData.visible_injury.display_message && (
                  <div className="p-3 bg-slate-950/40 border border-slate-700 text-slate-300 rounded-xl text-xs">
                    <strong>Image provenance: </strong> {caseData.visible_injury.display_message}
                  </div>
                )}
                {/* View Mode Selector Tabs */}
                <div className="flex items-center space-x-2 pb-1">
                  <button
                    type="button"
                    onClick={() => setImageTab("heatmap")}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                      imageTab === "heatmap"
                        ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm"
                        : "bg-slate-800/40 text-slate-400 border border-slate-800 hover:text-slate-200"
                    }`}
                  >
                    Grad-CAM (model visualization)
                  </button>
                  <button
                    type="button"
                    onClick={() => setImageTab("mask")}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                      imageTab === "mask"
                        ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm"
                        : "bg-slate-800/40 text-slate-400 border border-slate-800 hover:text-slate-200"
                    }`}
                  >
                    View Segmentation Mask
                  </button>
                  <button
                    type="button"
                    onClick={() => setImageTab("original")}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                      imageTab === "original"
                        ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm"
                        : "bg-slate-800/40 text-slate-400 border border-slate-800 hover:text-slate-200"
                    }`}
                  >
                    Original Photo
                  </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Visual Canvas — sized to exact overlay dimensions for coordinate fidelity */}
                <div
                  className="relative bg-slate-950 rounded-xl overflow-hidden border border-slate-800 mx-auto"
                  style={{
                    width: "100%",
                    maxWidth: "480px",
                    aspectRatio:
                      (caseData.visible_injury.original_width && caseData.visible_injury.original_height)
                        ? `${caseData.visible_injury.original_width} / ${caseData.visible_injury.original_height}`
                        : caseData.visible_injury.overlay_width && caseData.visible_injury.overlay_height
                        ? `${caseData.visible_injury.overlay_width} / ${caseData.visible_injury.overlay_height}`
                        : "1 / 1",
                  }}
                >
                  <img
                    src={
                      imageTab === "heatmap" && caseData.visible_injury.gradcam_overlay_generated && caseData.visible_injury.overlay_url
                        ? getUploadUrl(caseData.visible_injury.overlay_url)
                        : imageTab === "mask"
                        ? getUploadUrl(caseData.visible_injury.mask_url || `${caseData.case_id}_mask.png`)
                        : getUploadUrl(caseData.visible_injury.image_url || `${caseData.case_id}.jpg`)
                    }
                    alt={
                      imageTab === "heatmap"
                        ? "MODEL VISUALIZATION — NOT CLINICAL EXPLANATION"
                        : imageTab === "mask"
                        ? "U-Net Binary Segmentation Mask"
                        : "Original uploaded photo"
                    }
                    className="absolute inset-0 w-full h-full object-contain transition-opacity duration-300"
                    onError={(e) => {
                      const target = e.target as HTMLImageElement;
                      target.src = getUploadUrl(caseData.visible_injury!.image_url || `${caseData.case_id}.jpg`);
                    }}
                  />
                  
                  {/* Bounding Box — coords are original-image xyxy; container aspect matches original dims */}
                  {caseData.visible_injury.bounding_box &&
                    caseData.visible_injury.yolo_finding_detected === true &&
                    imageTab !== "mask" && (() => {
                      const box = caseData.visible_injury.bounding_box as number[];
                      const refW =
                        caseData.visible_injury.original_width ||
                        caseData.visible_injury.overlay_width ||
                        224;
                      const refH =
                        caseData.visible_injury.original_height ||
                        caseData.visible_injury.overlay_height ||
                        224;
                      return (
                    <div
                      className="absolute border-2 border-red-500 bg-red-500/15 rounded pointer-events-none"
                      style={{
                        left: `${(box[0] / refW) * 100}%`,
                        top: `${(box[1] / refH) * 100}%`,
                        width: `${Math.min(100, Math.max(0, ((box[2] - box[0]) / refW) * 100))}%`,
                        height: `${Math.min(100, Math.max(0, ((box[3] - box[1]) / refH) * 100))}%`,
                      }}
                    >
                      <span className="absolute top-0 left-0 bg-red-500 text-white text-[9px] px-1 rounded-br font-bold">
                        YOLO: {caseData.visible_injury.yolo_finding} ({((caseData.visible_injury.yolo_confidence || 0) * 100).toFixed(0)}%)
                      </span>
                    </div>
                      );
                    })()}

                  {/* No detection message */}
                  {caseData.visible_injury.yolo_finding_detected === false && imageTab !== "mask" && (
                    <div className="absolute bottom-2 left-2 right-2 bg-slate-900/80 backdrop-blur-sm rounded-lg text-[10px] text-amber-400 px-2 py-1 text-center border border-amber-900/40">
                      YOLO11: No confident injury detection (supported: {yoloSupportedClasses.join(", ") || "model.names only"})
                    </div>
                  )}

                  {imageTab === "heatmap" && caseData.visible_injury.gradcam_overlay_generated && (
                    <div className="absolute top-2 right-2 bg-slate-900/80 backdrop-blur-md px-2 py-0.5 rounded text-[10px] font-mono text-amber-300 border border-amber-500/30">
                      MODEL VISUALIZATION
                    </div>
                  )}
                  {imageTab === "heatmap" && !caseData.visible_injury.gradcam_overlay_generated && (
                    <div className="absolute inset-x-2 bottom-2 bg-slate-900/85 rounded-lg text-[10px] text-amber-300 px-2 py-1.5 border border-amber-800/50">
                      Grad-CAM withheld ({caseData.visible_injury.gradcam_explanation_status || "WITHHELD"}).
                      NOT CLINICAL EXPLANATION. Classifier status: {caseData.visible_injury.gradcam_model_status || caseData.visible_injury.classifier_status || "unavailable"}.
                    </div>
                  )}
                  {imageTab === "mask" && (
                    <div className="absolute top-2 right-2 bg-slate-900/80 backdrop-blur-md px-2 py-0.5 rounded text-[10px] font-mono border border-purple-500/30 text-purple-300">
                      {caseData.visible_injury.segmentation_reliable === false
                        ? `U-Net Mask Withheld (${caseData.visible_injury.segmentation_status || "UNRELIABLE"})`
                        : "U-Net Mask Active"}
                    </div>
                  )}
                  {imageTab === "mask" && caseData.visible_injury.segmentation_reliable === false && (
                    <div className="absolute inset-x-2 bottom-2 bg-slate-900/85 rounded-lg text-[10px] text-amber-300 px-2 py-1.5 border border-amber-800/50">
                      Segmentation not trusted for display (blank/OOD or quality gate). Raw FP diagnostics may still be stored for research.
                    </div>
                  )}
                </div>

                <div className="space-y-4 flex flex-col justify-between">

                  {/* ── YOLO11 Object Detection ─────────────────────────────── */}
                  <div className="p-3 bg-slate-950/70 border border-slate-700 rounded-xl space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-400">YOLO11 Object Detection</span>
                      <span className={`text-[9px] px-1.5 py-0.5 rounded font-bold ${
                        caseData.visible_injury.yolo_finding_detected
                          ? "bg-emerald-900/60 text-emerald-300 border border-emerald-700"
                          : "bg-amber-950/60 text-amber-400 border border-amber-800"
                      }`}>
                        {caseData.visible_injury.yolo_finding_detected ? "DETECTED" : "NO DETECTION"}
                      </span>
                    </div>
                    {/* NOTE: rest of YOLO/EfficientNet/UNet cards continue below — content was already gated by isTab("image") */}
                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Finding:</span>
                        <span className="font-semibold text-slate-200">
                          {caseData.visible_injury.yolo_finding || <span className="text-amber-400 italic">None detected</span>}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Confidence:</span>
                        <span className="font-semibold text-slate-200">
                          {caseData.visible_injury.yolo_confidence != null
                            ? `${(caseData.visible_injury.yolo_confidence * 100).toFixed(1)}%`
                            : <span className="text-slate-500">N/A</span>}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Supported classes:</span>
                        <span className="text-slate-500 text-[10px]">
                          {yoloSupportedClasses.join(", ") || "none (from model.names)"}
                        </span>
                      </div>
                      {(caseData.visible_injury.yolo_unsupported_classes || []).length > 0 && (
                        <div className="flex justify-between gap-2">
                          <span className="text-slate-400 shrink-0">Unsupported / no labels:</span>
                          <span className="text-amber-400 text-[10px] text-right">
                            {(caseData.visible_injury.yolo_unsupported_classes || []).join(", ")} (architecture only — not validated)
                          </span>
                        </div>
                      )}
                      {caseData.visible_injury.yolo_class_support_status === "UNSUPPORTED" && (
                        <p className="text-[10px] text-amber-400 font-semibold">
                          Detected class is UNSUPPORTED (zero honest training labels). Research demo only — not a validated finding.
                        </p>
                      )}
                      {caseData.visible_injury.yolo_dataset_provenance && (
                        <p className="text-[10px] text-slate-500">
                          YOLO model training data (not this uploaded image): {caseData.visible_injury.yolo_dataset_provenance}
                        </p>
                      )}
                    </div>
                  </div>

                  {/* ── EfficientNet Research Classifier ─────────────────────── */}
                  <div className="p-3 bg-slate-950/40 border border-slate-800 rounded-xl space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-sky-400">Research Image Classifier</span>
                      <span className={`text-[9px] px-1.5 py-0.5 rounded font-bold border ${
                        String(caseData.visible_injury.classifier_model_status || "").includes("NOT_TRUSTWORTHY")
                          ? "bg-amber-950/60 text-amber-400 border-amber-800"
                          : "bg-sky-950/60 text-sky-400 border-sky-800"
                      }`}>
                        EfficientNetV2 · {caseData.visible_injury.classifier_model_status || "status unknown"}
                      </span>
                    </div>
                    <div className="space-y-2">
                      {Object.entries(caseData.visible_injury.classification || {}).map(([cls, prob]: [string, any]) => {
                        const isWinner = cls === caseData.visible_injury.classifier_finding;
                        const isYoloCovered = yoloSupportedLower.has(String(cls).toLowerCase());
                        const numericProb = typeof prob === "number" && Number.isFinite(prob) ? prob : null;
                        return (
                          <div key={cls} className="space-y-0.5">
                            <div className="flex justify-between text-xs font-medium">
                              <span className={isWinner ? "text-sky-300 font-bold" : "text-slate-400"}>
                                {cls}
                                {!isYoloCovered && (
                                  <span className="ml-1 text-[9px] text-amber-500 font-normal">(not in YOLO)</span>
                                )}
                              </span>
                              <span className={isWinner ? "text-sky-300 font-bold" : "text-slate-500"}>
                                {numericProb == null ? "withheld" : `${(numericProb * 100).toFixed(1)}%`}
                              </span>
                            </div>
                            <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                              <div
                                className={`h-full rounded-full ${isWinner ? "bg-sky-500" : "bg-slate-600"}`}
                                style={{ width: `${numericProb == null ? 0 : numericProb * 100}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    <div className="space-y-1 pt-1 border-t border-slate-800">
                      <p className="text-[10px] text-amber-400 font-semibold">
                        Model training status: {caseData.visible_injury.classifier_model_status || "unknown"}
                      </p>
                      {caseData.visible_injury.classifier_abstention_class && (
                        <p className="text-[10px] text-sky-300 font-semibold">
                          Abstention class: {caseData.visible_injury.classifier_abstention_class} (not an injury finding).
                        </p>
                      )}
                      <p className="text-[10px] text-slate-500">
                        This prediction gate: {caseData.visible_injury.classifier_status || "n/a"} (input quality / confidence, not training status).
                      </p>
                      <p className="text-[10px] text-slate-500">
                        Research classifier category only. Classes marked <span className="text-amber-500">(not in YOLO)</span> are not detectable by YOLO11.
                      </p>
                      {caseData.visible_injury.classifier_yolo_coverage === "NOT AVAILABLE" && (
                        <p className="text-[10px] text-amber-500/80 font-semibold">
                          ⚠ Top category ({caseData.visible_injury.classifier_finding}) is NOT a YOLO11 detection class.
                        </p>
                      )}
                      {!caseData.visible_injury.classifier_is_confident && (
                        <p className="text-[10px] text-amber-400/80 font-semibold">
                          ⚠ Low classifier confidence — result may be unreliable for this image.
                        </p>
                      )}
                    </div>
                  </div>

                  {/* ── Segmentation + Grad-CAM metadata ─────────────────────── */}
                  <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-xl space-y-2 text-xs text-slate-400">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-sky-400">Research Segmentation</span>
                      <span className={`text-[9px] px-1.5 py-0.5 rounded font-bold border ${
                        String(caseData.visible_injury.segmentation_model_status || "").includes("NOT_TRUSTWORTHY")
                          ? "bg-amber-950/60 text-amber-400 border-amber-800"
                          : "bg-sky-950/60 text-sky-400 border-sky-800"
                      }`}>
                        U-Net · {caseData.visible_injury.segmentation_model_status || "status unknown"}
                      </span>
                    </div>
                    <div>
                      <span className="font-bold text-slate-200 block mb-0.5">Affected Area:</span>
                      {!caseData.visible_injury.segmentation_reliable || caseData.visible_injury.affected_ratio == null ? (
                        <div className="space-y-1">
                          <div className="flex items-center space-x-2">
                            <span className="text-sm font-bold text-amber-400 font-mono">N/A</span>
                            <span className="inline-block px-2 py-0.5 bg-amber-950/40 text-amber-400 border border-amber-800/50 rounded text-[10px] font-semibold">
                              No reliable segmentation mask available
                            </span>
                          </div>
                          <p className="text-[11px] text-slate-400">
                            Prediction gate: {caseData.visible_injury.segmentation_status || caseData.visible_injury.segmentation_trust || "insufficient"}.
                            Reason: {caseData.visible_injury.segmentation_reason || caseData.visible_injury.segmentation_message || "No reliable segmentation mask available."}
                          </p>
                          <p className="text-[10px] text-amber-400 font-semibold">
                            Model training status: {caseData.visible_injury.segmentation_model_status || "unknown"}
                          </p>
                        </div>
                      ) : (
                        <div className="space-y-1">
                          <p className="text-sm font-extrabold text-emerald-400">
                            {(caseData.visible_injury.affected_ratio * 100).toFixed(2)}% <span className="text-xs font-normal text-slate-400">of {caseData.visible_injury.denominator_label || "detected region"}</span>
                          </p>
                          <p className="text-[10px] text-amber-400 font-semibold">
                            Model training status: {caseData.visible_injury.segmentation_model_status || "unknown"}
                          </p>
                        </div>
                      )}
                    </div>

                    <div className="text-[11px] text-slate-300 border-t border-slate-800/80 pt-1.5 space-y-1">
                      <p className="font-bold text-amber-300 tracking-wide">MODEL VISUALIZATION</p>
                      <p className="font-bold text-amber-300 tracking-wide">NOT CLINICAL EXPLANATION</p>
                      <p className="text-slate-400">
                        Source model: {caseData.visible_injury.gradcam_source_model || "EfficientNetV2"}
                      </p>
                      <p className="text-slate-400">
                        Predicted class: {caseData.visible_injury.gradcam_predicted_class || "withheld"}
                      </p>
                      <p className="text-slate-400">
                        Confidence:{" "}
                        {caseData.visible_injury.gradcam_confidence != null
                          ? `${(caseData.visible_injury.gradcam_confidence * 100).toFixed(1)}%`
                          : "N/A"}
                      </p>
                      <p className="text-slate-400">
                        Model status: {caseData.visible_injury.gradcam_model_status || caseData.visible_injury.classifier_status || "unknown"}
                      </p>
                      <p className="text-slate-400">
                        Explanation status: {caseData.visible_injury.gradcam_explanation_status || "WITHHELD"}
                      </p>
                      {!caseData.visible_injury.gradcam_overlay_generated && caseData.visible_injury.gradcam_withheld_reason && (
                        <p className="text-slate-500">
                          Withheld reason: {caseData.visible_injury.gradcam_withheld_reason}
                        </p>
                      )}
                    </div>
                    {caseData.visible_injury.overlay_width && (
                      <p className="text-[10px] text-slate-600 font-mono">
                        Overlay: {caseData.visible_injury.overlay_width}×{caseData.visible_injury.overlay_height}px (original dimensions)
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </div>
            ) : (
              <p className="text-sm text-slate-500 italic py-4">Image analysis skipped or unavailable.</p>
            )}
          </div>
        </div>
      </div>
      )}

          {/* Section 2: Questionnaire */}
          {isTab("overview", "questionnaire") && (
          <div id="case-panel-questionnaire" role="tabpanel" aria-labelledby="case-tab-questionnaire" className="bg-slate-900/50 border border-slate-800 p-6 rounded-2xl space-y-4">
            <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
              <FileText className="h-5 w-5 text-emerald-400" />
              <h3 className="font-bold text-white text-base">Section 2: Injury Questionnaire Context</h3>
            </div>

            {caseData.questionnaire ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {Object.entries(caseData.questionnaire.answers || {}).map(([key, val]) => {
                    const isNotProvided = val == null || String(val).trim() === "" || String(val).toLowerCase() === "not_provided";
                    return (
                      <div key={key} className="p-3 bg-slate-800/20 border border-slate-850 rounded-xl space-y-1">
                        <span className="text-[10px] text-slate-500 uppercase font-bold block">
                          {key.replace(/_/g, " ")}
                        </span>
                        <span className="text-sm font-semibold text-slate-200 block capitalize">
                          {isNotProvided ? (
                            <span className="text-slate-500 italic font-normal">Not provided</span>
                          ) : (
                            key === "pain_level" ? `${val}/10` : String(val)
                          )}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <p className="text-sm text-slate-500 italic py-4">Questionnaire answers unavailable.</p>
            )}
          </div>
          )}

          {/* Section 3: Sensor Processing */}
          {isTab("overview", "sensor") && (
          <div id="case-panel-sensor" role="tabpanel" aria-labelledby="case-tab-sensor">
          {caseData.sensor_summary && (
            <div className="bg-slate-900/50 border border-slate-800 p-6 rounded-2xl space-y-6">
              {/* Sensor provenance label */}
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Cpu className="h-5 w-5 text-emerald-400" />
                  <h3 className="font-bold text-white text-base">Section 3: Smartphone Sensor Log Features</h3>
                </div>
                {caseData.sensor_source_type && (
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded border font-mono ${
                    caseData.sensor_source_type === "live"
                      ? "bg-emerald-950/80 text-emerald-400 border-emerald-800"
                      : caseData.sensor_source_type === "uploaded"
                      ? "bg-emerald-950/40 text-emerald-400 border-emerald-900/40"
                      : "bg-purple-950/40 text-purple-400 border-purple-900/40"
                  }`}>
                    {caseData.sensor_source_type === "live" ? "📱 REAL-TIME DEVICE DATA" :
                     caseData.sensor_source_type === "uploaded" ? "USER UPLOADED DATA" :
                     caseData.sensor_source_type === "demo" ? "DEMO DATA" :
                     caseData.sensor_source_type === "simulated" ? "SIMULATED DATA" : caseData.sensor_source_type.toUpperCase()}
                  </span>
                )}
              </div>

              {caseData.sensor_source_type === "live" && (
                <div className="p-3.5 bg-emerald-950/20 border border-emerald-800/40 text-emerald-300 rounded-xl text-xs space-y-1">
                  <div className="flex justify-between items-center">
                    <strong className="font-bold text-emerald-400">Real-Time Mobile Device Capture Metadata</strong>
                    <span className="text-[10px] font-mono bg-emerald-900/80 text-emerald-200 px-1.5 py-0.5 rounded">VERIFIED BY BACKEND</span>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 font-mono text-[11px] text-slate-300">
                    <div>Duration: <strong className="text-white">{caseData.sensor_summary?.recording_duration_seconds || "N/A"}s</strong></div>
                    <div>Samples: <strong className="text-white">{caseData.sensor_summary?.sample_count || "N/A"}</strong></div>
                    <div>Rate: <strong className="text-white">{caseData.sensor_summary?.backend_verified_sampling_rate_hz || "N/A"} Hz</strong></div>
                    <div>Unit: <strong className="text-white">m/s²</strong></div>
                  </div>
                </div>
              )}

              {caseData.sensor_summary?.source_type && ["demo", "simulated"].includes(caseData.sensor_summary.source_type) && (
                <div className="p-3 bg-purple-950/20 border border-purple-850 text-purple-300 rounded-xl text-xs space-y-1">
                  <strong>Demo Mode: </strong> Synthetic sensor data is used for demonstration only. It does not represent a real person&apos;s accident.

                  {caseData.sensor_summary?.data_provenance_message && (
                    <p className="text-slate-400 mt-1 italic">{caseData.sensor_summary.data_provenance_message}</p>
                  )}
                </div>
              )}
              <div className="border-b border-slate-800 pb-3" />

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-slate-800/30 border border-slate-850 rounded-xl text-center">
                      <span className="block text-2xl font-extrabold text-white">
                        {caseData.sensor_summary.peak_g_force == null
                          ? "FEATURE_MISSING"
                          : `${Number(caseData.sensor_summary.peak_g_force).toFixed(2)}g`}
                      </span>
                      <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">Peak Impact Intensity</span>
                    </div>
                    <div className="p-4 bg-slate-800/30 border border-slate-850 rounded-xl text-center">
                      <span className="block text-sm font-extrabold text-white break-all">
                        {caseData.sensor_summary.predicted_motion_class
                          || caseData.sensor_summary.motion_classification?.predicted_motion_class
                          || caseData.sensor_summary.classifier_status
                          || "FEATURE_MISSING"}
                      </span>
                      <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">Motion Classifier</span>
                      {typeof caseData.sensor_summary.motion_confidence === "number" && (
                        <span className="block text-[10px] text-slate-400 font-mono mt-1">
                          {(caseData.sensor_summary.motion_confidence * 100).toFixed(1)}%
                        </span>
                      )}
                      {caseData.sensor_summary.classifier_status
                        && caseData.sensor_summary.classifier_status !== (caseData.sensor_summary.predicted_motion_class
                          || caseData.sensor_summary.motion_classification?.predicted_motion_class) && (
                        <span className="block text-[10px] text-slate-500 font-mono mt-1">
                          {caseData.sensor_summary.classifier_status}
                        </span>
                      )}
                    </div>
                    <div className="p-4 bg-slate-800/30 border border-slate-850 rounded-xl text-center">
                      <span className="block text-2xl font-extrabold text-white">
                        {caseData.sensor_summary.post_impact_stabilization_seconds == null
                          ? "FEATURE_MISSING"
                          : `${Number(caseData.sensor_summary.post_impact_stabilization_seconds).toFixed(2)}s`}
                      </span>
                      <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">Stabilization Duration</span>
                    </div>
                  </div>
                  {Array.isArray(caseData.sensor_summary.motion_classification?.missing_features)
                    && caseData.sensor_summary.motion_classification.missing_features.length > 0 && (
                    <p className="text-[10px] text-amber-400 font-mono">
                      FEATURE_MISSING: {caseData.sensor_summary.motion_classification.missing_features.join(", ")}
                    </p>
                  )}
                </div>

                <div className="space-y-2">
                  <span className="text-xs font-bold text-slate-500 uppercase tracking-wider block">Accident Timeline Reconstruction</span>
                  <div className="relative pl-6 border-l border-slate-800 space-y-4 text-xs">
                    {(caseData.sensor_summary.events && caseData.sensor_summary.events.length > 0
                      ? caseData.sensor_summary.events
                      : [{ time_offset_seconds: 0, event_name: "Sensor log", description: "No timeline events recorded." }]
                    ).map((ev: any, idx: number) => (
                      <div key={idx} className="relative">
                        <div className="absolute -left-[30px] top-1 h-2 w-2 rounded-full bg-slate-500" />
                        <p className="text-white font-semibold">{ev.event_name}</p>
                        {ev.description && (
                          <p className="text-slate-400 text-[11px]">{ev.description}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
          {!caseData.sensor_summary && (
            <div className="bg-slate-900/50 border border-slate-800 p-6 rounded-2xl space-y-3">
              <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
                <Cpu className="h-5 w-5 text-slate-500" />
                <h3 className="font-bold text-slate-400 text-base">Section 3: Smartphone Sensor Log Features</h3>
              </div>
              <div className="p-4 bg-slate-950/40 border border-slate-850 rounded-xl text-xs text-slate-400">
                <strong>Sensor Data Status:</strong> Sensor data not provided (skipped or omitted). Reduced-modality model configuration applied automatically.
              </div>
            </div>
          )}
          </div>
          )}

        {/* RIGHT COLUMN: AI/ML Models Fusion, Predictions & Safety */}
        {isTab("overview", "ai") && (
        <div id="case-panel-ai" role="tabpanel" aria-labelledby="case-tab-ai" className="space-y-8">
          
          {/* Section 4: Multimodal Analysis Consistency */}
          <div className="bg-slate-900/50 border border-slate-800 p-6 rounded-2xl space-y-4">
            <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
              <Activity className="h-5 w-5 text-emerald-400" />
              <h3 className="font-bold text-white text-base">Section 4: Multimodal Fusion & Model Agreement</h3>
            </div>
            
            <div className="p-3 bg-amber-950/30 border border-amber-800/50 rounded-xl text-xs text-amber-200 space-y-1">
              <strong className="block text-amber-400">CLINICAL CLAIM BLOCKED</strong>
              <p>
                XGBoost and VQC training labels are synthetic rule-derived categories, not paired clinician labels.
                There are 0 genuinely paired clinical multimodal records in this project. These outputs must not be cited as medical triage.
              </p>
              {(caseData.xgboost_prediction?.clinical_claim || caseData.fusion_label_source) && (
                <p className="font-mono text-[10px] text-slate-400">
                  {caseData.xgboost_prediction?.clinical_claim || "BLOCKED_NO_PAIRED_CLINICAL_LABELS"}
                  {caseData.xgboost_prediction?.label_source
                    ? ` · labels=${caseData.xgboost_prediction.label_source}`
                    : ""}
                  {typeof caseData.xgboost_prediction?.paired_clinical_samples === "number"
                    ? ` · paired_clinical_samples=${caseData.xgboost_prediction.paired_clinical_samples}`
                    : ""}
                </p>
              )}
            </div>
            <div className="p-3 bg-purple-950/20 border border-purple-850/50 rounded-xl text-xs text-purple-300">
              <strong>Notice on Output Separation: </strong>
              Model predictions, rule-derived safety guidance, and first-aid guidance are separate outputs and should not be interpreted as a single clinical classification.
            </div>

            <div className="space-y-3 text-xs">
                <div className="flex justify-between items-center py-2 border-b border-slate-850">
                  <span className="text-slate-400">Modalities Used:</span>
                  <span className="font-mono text-slate-200">
                    {(caseData as any).modalities_used?.join(" + ") || (caseData.sensor_available ? "Image + Questionnaire + Sensor" : "Image + Questionnaire")}
                  </span>
                </div>
                {(caseData as any).model_configuration_used && (
                  <div className="flex justify-between items-center py-2 border-b border-slate-850">
                    <span className="text-slate-400">Model Config:</span>
                    <span className="font-mono text-slate-300 text-[10px]">{(caseData as any).model_configuration_used}</span>
                  </div>
                )}
                <div className="flex justify-between items-center py-2 border-b border-slate-850">
                  <span className="text-slate-400">Model Agreement:</span>
                  <span className={`font-bold px-2 py-0.5 rounded text-[11px] ${
                    caseData.prediction_agreement === "AGREEMENT" || caseData.agreement_score === "AGREEMENT"
                      ? "bg-emerald-950/60 text-emerald-400 border border-emerald-800"
                      : "bg-red-950/60 text-red-400 border border-red-800"
                  }`}>
                    {caseData.prediction_agreement || caseData.agreement_score || "AGREEMENT"}
                  </span>
                </div>
                {(caseData.prediction_agreement === "DISAGREEMENT" || caseData.agreement_score === "DISAGREEMENT") && (
                  <div className="p-3 bg-red-950/20 border border-red-800/40 rounded-xl space-y-1 text-xs text-red-300">
                    <span className="font-bold block text-red-400">Model Prediction Disagreement:</span>
                    <p className="text-[11px] text-slate-300">
                      Classical XGBoost predicted <strong className="text-emerald-400">{caseData.xgboost_prediction?.class || "N/A"}</strong> while Experimental 4-Qubit VQC predicted <strong className="text-purple-400">{caseData.quantum_prediction?.class || "N/A"}</strong>.
                    </p>
                  </div>
                )}
                {(caseData as any).consistency_analysis && (
                  <div className="flex justify-between items-center py-2 border-b border-slate-850">
                    <span className="text-slate-400">Multimodal Evidence Consistency:</span>
                    <span className="font-semibold text-slate-300">
                      {(caseData as any).consistency_analysis.score}% <span className="text-[10px] text-slate-500 font-mono">({(caseData as any).consistency_analysis.status})</span>
                    </span>
                  </div>
                )}
                <div className="flex justify-between items-center py-2 border-b border-slate-850">
                  <span className="text-slate-400">Research Uncertainty Level:</span>
                  <span className={`font-semibold ${
                    (caseData.uncertainty_level || caseData.uncertainty_status) === "LOW UNCERTAINTY" || caseData.uncertainty_status === "High Certainty"
                      ? "text-emerald-400"
                      : "text-amber-400 font-bold"
                  }`}>
                    {caseData.uncertainty_level || caseData.uncertainty_status || "LOW UNCERTAINTY"}
                  </span>
                </div>
                {caseData.uncertainty_reasons && caseData.uncertainty_reasons.length > 0 && (
                  <div className="p-3 bg-slate-950/40 border border-slate-800 rounded-xl space-y-1 text-xs text-slate-300">
                    <span className="font-bold text-slate-400 block text-[10px] uppercase tracking-wider">Uncertainty Drivers:</span>
                    <ul className="list-disc list-inside text-[11px] text-amber-400 space-y-0.5">
                      {caseData.uncertainty_reasons.map((r: string, idx: number) => (
                        <li key={idx}>{r}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
          </div>


          {/* Section 5: Classical Research Model Prediction (XGBoost) */}
          {caseData.xgboost_prediction && (
            <div className="bg-slate-900/50 border border-slate-800 p-6 rounded-2xl space-y-4">
              <div className="flex justify-between items-center border-b border-slate-800 pb-3">
                <h3 className="font-bold text-white text-base">Section 5: Classical Research Model Prediction</h3>
                <span className="text-[10px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded font-mono">XGBoost Classifier</span>
              </div>

              <div className="text-center p-4 bg-slate-950/40 rounded-xl border border-slate-850 space-y-2">
                <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wider block">Rule-Derived Research Category</span>
                <span className="text-3xl font-extrabold bg-gradient-to-r from-emerald-400 to-emerald-300 bg-clip-text text-transparent tracking-wide">
                  {caseData.xgboost_prediction.class}
                </span>
                <span className="block text-[10px] text-slate-400 italic">
                  Not a clinical severity classification. XGBoost learns predefined research labels derived from multimodal rules.
                </span>
                <span className="block text-xs text-slate-400">
                  Model probability: {(caseData.xgboost_prediction.probability * 100).toFixed(0)}%
                </span>
                {caseData.xgboost_prediction.data_provenance && (
                  <span className="block text-[10px] text-slate-500 font-mono uppercase tracking-wider">
                    Training data: {caseData.xgboost_prediction.data_provenance}
                    {caseData.xgboost_prediction.n_features
                      ? ` · ${caseData.xgboost_prediction.n_features}-feature schema`
                      : ""}
                  </span>
                )}
              </div>


              {/* SHAP explanation */}
              {caseData.shap_explanations && (
                <div className="space-y-2 text-xs">
                  <span className="font-bold text-slate-500 uppercase tracking-wider block">SHAP Local Feature Contribution</span>
                  <div className="space-y-2 pt-2 max-h-48 overflow-y-auto pr-1">
                    {caseData.shap_explanations.slice(0, 5).map((exp: any, i: number) => {
                      const absoluteVal = Math.abs(exp.shap_value);
                      const isPositive = exp.shap_value >= 0;
                      return (
                        <div key={i} className="space-y-1">
                          <div className="flex justify-between text-[11px] text-slate-300">
                            <span className="capitalize">{exp.feature.replace("_", " ")}</span>
                            <span className={isPositive ? "text-red-400" : "text-emerald-400"}>
                              {isPositive ? "+" : ""}{exp.shap_value.toFixed(3)}
                            </span>
                          </div>
                          <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                            <div 
                              className={`h-full ${isPositive ? "bg-red-400" : "bg-emerald-500"}`} 
                              style={{ width: `${Math.min(100, absoluteVal * 200)}%` }} 
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <span className="text-[9px] text-slate-500 block italic leading-normal">
                    SHAP analyzes input contributions to predicted classes and does not provide clinical diagnostic verification.
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Section 6: Experimental Quantum Model Prediction (VQC) */}
          {caseData.quantum_prediction && (
            <div className="bg-slate-900/50 border border-slate-800 p-6 rounded-2xl space-y-4">
              <div className="flex justify-between items-center border-b border-slate-800 pb-3">
                <h3 className="font-bold text-white text-base">Section 6: Experimental Quantum Model Prediction</h3>
                <span className="text-[10px] bg-amber-950/40 text-amber-400 px-2 py-0.5 rounded border border-amber-900/40 font-mono">EXPERIMENTAL_ONLY</span>
              </div>

              <div className="text-center p-4 bg-slate-950/40 rounded-xl border border-slate-850 space-y-1">
                <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wider block">EXPERIMENTAL VQC PREDICTION</span>
                <span className="text-2xl font-extrabold text-emerald-400 tracking-wide">
                  {caseData.quantum_prediction.status === "MODEL_UNAVAILABLE"
                    ? "MODEL UNAVAILABLE"
                    : (caseData.quantum_prediction.class ?? "EXPERIMENTAL")}
                </span>
                <span className="block text-[10px] text-slate-500 mt-1 font-mono">
                  {Array.isArray(caseData.quantum_prediction.score)
                    ? `Experimental VQC outputs: [${caseData.quantum_prediction.score.map((s: number) => s.toFixed(3)).join(", ")}]`
                    : (caseData.quantum_prediction.error
                      ? `VQC failed: ${caseData.quantum_prediction.error}`
                      : "VQC scores unavailable — no fallback probabilities")}
                </span>
                <span className="block text-[10px] text-amber-400/90 mt-2 font-mono">
                  Isolated from main decision ({caseData.quantum_prediction.status || "EXPERIMENTAL_ONLY"})
                  {caseData.quantum_prediction.data_provenance
                    ? ` · training data: ${caseData.quantum_prediction.data_provenance}`
                    : ""}
                </span>
              </div>
              
              <div className="p-3 bg-slate-950/40 rounded-xl border border-slate-850 space-y-2 text-xs text-slate-400 font-mono">
                <div className="grid grid-cols-2 gap-2 text-[11px] pb-2 border-b border-slate-800">
                  <div><strong>Simulator:</strong> PennyLane default.qubit</div>
                  <div><strong>Qubits:</strong> 4</div>
                  <div><strong>Encoding:</strong> Angle Embedding</div>
                  <div><strong>Input:</strong> 4 PCA components</div>
                  <div className="col-span-2 text-amber-400 font-bold">Quantum Advantage: NOT CLAIMED</div>
                </div>
                <p className="text-[10px] text-slate-400 font-sans italic">
                  Executed on PennyLane default.qubit simulator. Quantum advantage not claimed.
                </p>
              </div>
            </div>
          )}


          {/* Section 8: Rule-Derived Research Safety Guidance */}
          {caseData.safety_information && (
            <div className="bg-slate-900/50 border border-slate-800 p-6 rounded-2xl space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center space-x-2">
                  <LifeBuoy className="h-5 w-5 text-emerald-400" />
                  <h3 className="font-bold text-white text-base">Section 8: Rule-Derived Research Safety Guidance</h3>
                </div>
                <span className="text-[10px] bg-amber-950/40 text-amber-400 border border-amber-850 px-2 py-0.5 rounded font-mono font-bold">
                  SAFETY GUIDANCE LEVEL: {caseData.safety_guidance_level || caseData.rule_derived_category || caseData.safety?.rule_derived_category || "MODERATE"}
                </span>
              </div>

              <div className="p-3 bg-slate-950/40 border border-slate-850 rounded-xl space-y-1 text-[11px] text-slate-400">
                <p>
                  <strong>Rule-Derived Research Safety Guidance:</strong> Generated from predefined safety rules using available evidence.
                </p>
                <p className="text-[10px] text-amber-400/90 font-mono">
                  This is generated from predefined safety rules and is separate from ML model predictions. Not a clinical severity classification.
                </p>
              </div>

              <ul className="space-y-2 text-xs text-slate-300">
                {caseData.safety_information.map((info: string, idx: number) => (
                  <li key={idx} className="flex items-start space-x-2">
                    <span className="text-emerald-400 font-bold mt-0.5">&bull;</span>
                    <span>{info}</span>
                  </li>
                ))}
              </ul>
              
              {/* Enforce Fracture and Prototype disclaimers */}
              <div className="pt-2 border-t border-slate-800 space-y-2 text-[10px] text-amber-400 leading-normal">
                <p>* Predefined safety reference text; research prototype only — no real emergency services contacted.</p>
                {(caseData.questionnaire?.answers?.crack_pop === "yes" || caseData.xgboost_prediction?.class !== "LOW") && (
                  <p className="font-semibold text-red-400">
                    * An ordinary RGB photograph cannot reliably determine a fracture. Appropriate medical imaging and professional assessment are required.
                  </p>
                )}
              </div>
            </div>
          )}

        </div>
        )}

      {/* Section 9: Research Prototype — Basic First-Aid Guidance */}
      {isTab("overview", "report") && (caseData.first_aid_guidance || caseData.report?.first_aid_guidance) && (() => {
        const fa = caseData.first_aid_guidance || caseData.report?.first_aid_guidance;
        const isGemini = fa.provider === "gemini" && fa.status === "success";
        return (
          <div id="case-panel-report-aid" className="bg-slate-900/50 border border-slate-800 p-6 rounded-2xl space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-slate-800 pb-3 gap-2">
              <div className="flex items-center space-x-2">
                <HeartPulse className="h-5 w-5 text-emerald-400 shrink-0" />
                <h3 className="font-bold text-white text-base">Research Prototype — Basic First-Aid Guidance</h3>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <span className={`text-[10px] px-2 py-0.5 rounded font-mono font-bold border ${
                  isGemini
                    ? "bg-emerald-950/60 text-emerald-300 border-emerald-700"
                    : "bg-amber-950/60 text-amber-300 border-amber-800"
                }`}>
                  Provider: {isGemini ? "Google Gemini (gemini-2.5-flash)" : "Rule-Based Fallback"}
                </span>
                <span className="text-[10px] bg-slate-800 text-slate-300 border border-slate-700 px-2 py-0.5 rounded font-mono font-bold">
                  {fa.guidance_level || "MODERATE"}
                </span>
              </div>
            </div>

            {/* Banner notice based on provider status */}
            {!isGemini ? (
              <div className="p-3 bg-amber-950/30 border border-amber-800/40 rounded-xl text-xs text-amber-300 flex items-start space-x-2">
                <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold block">AI-generated guidance unavailable. Showing rule-based research guidance.</span>
                  <span className="text-[11px] text-slate-400 block mt-0.5 break-words">
                    {fa.fallback_reason ? `Reason: ${fa.fallback_reason}` : "Loaded deterministic fallback rules derived from structured evidence."}
                  </span>
                </div>
              </div>
            ) : (
              <div className="p-3 bg-emerald-950/30 border border-emerald-800/40 rounded-xl text-xs text-emerald-300 flex items-start space-x-2">
                <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold block">Generated via Gemini (gemini-2.5-flash)</span>
                  <span className="text-[11px] text-slate-300 block mt-0.5 break-words">
                    Synthesized from canonical StructuredEvidence object (Hash: {fa.evidence_hash || "N/A"}). Non-diagnostic research prototype output.
                  </span>
                </div>
              </div>
            )}

            {/* Evidence Summary */}
            {fa.evidence_summary?.length > 0 && (
              <div className="p-3.5 bg-slate-950/60 border border-slate-800 rounded-xl space-y-2 text-xs">
                <h4 className="font-bold text-slate-200 flex items-center gap-1.5 text-[12px]">
                  <span>🔍</span> Structured Evidence Used
                </h4>
                <ul className="space-y-1 text-slate-300 list-disc list-inside text-[11px]">
                  {fa.evidence_summary.map((ev: string, idx: number) => (
                    <li key={idx} className="break-words">{ev}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Urgent Warning Signs */}
            {fa.urgent_warning_signs?.length > 0 && (
              <div className="p-3 bg-red-950/30 border border-red-800/40 rounded-xl space-y-1.5 text-xs text-red-300">
                <div className="flex items-center space-x-2 font-bold text-red-400">
                  <AlertTriangle className="h-4 w-4 shrink-0" />
                  <span>Urgent Medical Evaluation Warning Signs:</span>
                </div>
                <ul className="list-disc list-inside space-y-1 text-[11px]">
                  {fa.urgent_warning_signs.map((ws: string, idx: number) => (
                    <li key={idx} className="break-words">{ws}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Guidance Sections — Responsive Grid (1 col on narrow, 2 cols on medium, 3 cols on wide) */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
              <div className="bg-slate-950/40 p-4 rounded-xl border border-slate-850 space-y-2">
                <h4 className="font-bold text-emerald-400 flex items-center gap-1.5">
                  <span>✓</span> Immediate First-Aid Steps
                </h4>
                <ul className="space-y-1.5 text-slate-300 list-disc list-inside text-[11px] leading-relaxed">
                  {(fa.guidance?.immediate_first_aid_steps || fa.immediate_steps || []).map((step: string, idx: number) => (
                    <li key={idx} className="break-words">{step}</li>
                  ))}
                </ul>
              </div>

              <div className="bg-slate-950/40 p-4 rounded-xl border border-slate-850 space-y-2">
                <h4 className="font-bold text-amber-400 flex items-center gap-1.5">
                  <span>⚠</span> Actions to Avoid
                </h4>
                <ul className="space-y-1.5 text-slate-300 list-disc list-inside text-[11px] leading-relaxed">
                  {(fa.guidance?.actions_to_avoid || fa.avoid || []).map((item: string, idx: number) => (
                    <li key={idx} className="break-words">{item}</li>
                  ))}
                </ul>
              </div>

              <div className="bg-slate-950/40 p-4 rounded-xl border border-slate-850 space-y-2">
                <h4 className="font-bold text-sky-400 flex items-center gap-1.5">
                  <span>👁</span> Symptoms to Monitor
                </h4>
                <ul className="space-y-1.5 text-slate-300 list-disc list-inside text-[11px] leading-relaxed">
                  {(fa.guidance?.symptoms_to_monitor || fa.monitor || []).map((item: string, idx: number) => (
                    <li key={idx} className="break-words">{item}</li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="pt-2 border-t border-slate-800 text-[10px] text-slate-400 space-y-1 italic break-words">
              <p>
                <strong>Professional Evaluation Guidance:</strong> {fa.professional_evaluation_warning || fa.guidance?.professional_evaluation_guidance?.join(" ") || "Seek medical evaluation if symptoms worsen."}
              </p>
              <p className="text-amber-400/90">
                * AI-QTriage is an academic research prototype and does not provide clinical medical diagnosis.
              </p>
            </div>
          </div>
        );
      })()}

      {/* Counterfactual Sensitivity sweeps */}
      {isTab("overview", "explainability") && caseData.counterfactual_analysis && (
        <div id="case-panel-explainability" role="tabpanel" aria-labelledby="case-tab-explainability" className="p-6 bg-slate-900 border border-slate-800 rounded-2xl space-y-6">
          <div className="border-b border-slate-800 pb-4">
            <h3 className="font-bold text-lg text-white flex items-center gap-2">
              <Sliders className="h-5 w-5 text-emerald-400" />
              Counterfactual Decision Boundary Sensitivity Sweeps
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              {caseData.counterfactual_analysis.explanation}
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs">
            {/* Pain level sweeps */}
            <div className="bg-slate-950/40 p-4 border border-slate-850 rounded-xl space-y-3">
              <h4 className="font-bold text-slate-200">Pain Scale Sweep Sensitivity (0 to 10)</h4>
              {caseData.counterfactual_analysis.pain_sensitivity.classical_xgb_transitions && 
               caseData.counterfactual_analysis.pain_sensitivity.classical_xgb_transitions.length > 0 ? (
                <ul className="space-y-1.5 text-slate-350 list-disc list-inside">
                  {caseData.counterfactual_analysis.pain_sensitivity.classical_xgb_transitions.map((tr: any, idx: number) => (
                    <li key={idx}>
                      At pain level <strong className="text-slate-200">{tr.pain_level}/10</strong>, the classical model toggles output to <strong className="text-emerald-400">{tr.new_prediction}</strong>.
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-slate-500 italic">No category transitions found within pain range sweeps.</p>
              )}
            </div>

            {/* G-Force sweeps */}
            <div className="bg-slate-950/40 p-4 border border-slate-850 rounded-xl space-y-3">
              <h4 className="font-bold text-slate-200">Peak G-Force Sweep Sensitivity (1.0g to 8.0g)</h4>
              {caseData.counterfactual_analysis.g_force_sensitivity.classical_xgb_transitions &&
               caseData.counterfactual_analysis.g_force_sensitivity.classical_xgb_transitions.length > 0 ? (
                <ul className="space-y-1.5 text-slate-350 list-disc list-inside">
                  {caseData.counterfactual_analysis.g_force_sensitivity.classical_xgb_transitions.map((tr: any, idx: number) => (
                    <li key={idx}>
                      At impact spike <strong className="text-slate-200">{tr.peak_g_force}g</strong>, the model prediction changes to <strong className="text-emerald-400">{tr.new_prediction}</strong>.
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-slate-500 italic">No category transitions found within g-force sweeps.</p>
              )}
            </div>
          </div>
        </div>
      )}
      {isTab("overview", "explainability") && !caseData.counterfactual_analysis && (
        <div id="case-panel-explainability" role="tabpanel" aria-labelledby="case-tab-explainability" className="p-6 bg-slate-900 border border-slate-800 rounded-2xl space-y-2">
          <h3 className="font-bold text-white flex items-center gap-2">
            <Sliders className="h-5 w-5 text-emerald-400" />
            Explainability
          </h3>
          <p className="text-xs text-slate-400">
            Grad-CAM overlays are in Image analysis above. SHAP feature contributions are with XGBoost in AI / Quantum.
            Counterfactual sweeps were not stored for this case.
          </p>
        </div>
      )}

      {/* Accordion: Technical Details for Judges/Faculty */}
      {isTab("overview", "report") && (
      <div id="case-panel-report" role="tabpanel" aria-labelledby="case-tab-report" className="bg-slate-950/50 border border-slate-850 rounded-2xl overflow-hidden">
        <button
          onClick={() => setShowTechnical(!showTechnical)}
          className="w-full p-6 flex justify-between items-center text-sm font-bold text-slate-300 hover:bg-slate-900/40 transition-colors"
        >
          <span className="flex items-center space-x-2">
            <Cpu className="h-4 w-4 text-emerald-400" />
            <span>Show Research Details (Model Architectures & PDF/JSON Export)</span>
          </span>
          {showTechnical ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </button>

        {showTechnical && (
          <div className="p-6 border-t border-slate-850 bg-slate-900/20 space-y-6">
            
            {/* Export and download reports */}
            <div className="space-y-3">
              <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider">Export Case Assessment Reports</h4>
              <p className="text-xs text-slate-400">
                Download printable PDF or structured JSON files containing this research case&apos;s complete assessments.
              </p>
              <div className="flex flex-wrap gap-4 pt-2">
                <Link 
                  href={getApiUrl(`/api/cases/${caseData.case_id}/report/pdf`)}
                  target="_blank"
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-xs font-bold text-emerald-400 flex items-center gap-2 border border-slate-750 transition-colors"
                >
                  <Download className="h-3.5 w-3.5" />
                  Download printable PDF Report
                </Link>
                <Link 
                  href={getApiUrl(`/api/cases/${caseData.case_id}/report/json`)}
                  target="_blank"
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-xs font-bold text-emerald-400 flex items-center gap-2 border border-slate-750 transition-colors"
                >
                  <Download className="h-3.5 w-3.5" />
                  Download JSON Report
                </Link>
              </div>
            </div>

            {/* PCA features component list — live from this case's VQC projection, never hardcoded */}
            <div className="space-y-3 pt-2">
              <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider">Classical PCA-based Dimensionality Reduction</h4>
              <p className="text-xs text-slate-400">
                The fused 23-feature vector is compressed to 4 components by the train-fitted VQC scaler/PCA before the PennyLane circuit. Values below are from this case, or marked unavailable if VQC did not run.
              </p>
              {Array.isArray(caseData.quantum_prediction?.pca_features) && caseData.quantum_prediction.pca_features.length === 4 ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono bg-slate-950/40 p-4 border border-slate-850 rounded-xl">
                  {caseData.quantum_prediction.pca_features.map((value: number, idx: number) => (
                    <div key={`pc${idx + 1}`}>
                      <span className="block text-[10px] text-slate-500 font-bold">COMPONENT {idx + 1} (PC{idx + 1})</span>
                      <span className="text-slate-300">{Number(value).toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs font-mono bg-slate-950/40 p-4 border border-slate-850 rounded-xl text-amber-300">
                  PCA components not stored on this case. {caseData.quantum_prediction?.pca_note || "Re-run analysis to project the live fused vector."}
                </div>
              )}
            </div>

            {/* VQC circuit stubs */}
            <div className="space-y-3 pt-2">
              <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider">Variational Quantum Circuit (VQC) Structure</h4>
              <p className="text-xs text-slate-400">
                Qubit configuration and layers mapped classically via simulation:
              </p>
              <div className="p-4 bg-slate-950/40 border border-slate-850 rounded-xl font-mono text-[10px] text-emerald-400 space-y-1 overflow-x-auto">
                <p>Qubit 0: &mdash;&mdash;[AngleEmbedding(PC1)]&mdash;&mdash;[StronglyEntanglingLayer(W1)]&mdash;&mdash;[Measure PauliZ]&mdash;&mdash;</p>
                <p>Qubit 1: &mdash;&mdash;[AngleEmbedding(PC2)]&mdash;&mdash;[StronglyEntanglingLayer(W1)]&mdash;&mdash;[Measure PauliZ]&mdash;&mdash;</p>
                <p>Qubit 2: &mdash;&mdash;[AngleEmbedding(PC3)]&mdash;&mdash;[StronglyEntanglingLayer(W1)]&mdash;&mdash;[Measure PauliZ]&mdash;&mdash;</p>
                <p>Qubit 3: &mdash;&mdash;[AngleEmbedding(PC4)]&mdash;&mdash;[StronglyEntanglingLayer(W1)]&mdash;&mdash;[Measure PauliZ]&mdash;&mdash;</p>
              </div>
            </div>
            
          </div>
        )}
      </div>
      )}
      </div>
      
    </div>
  );
}
