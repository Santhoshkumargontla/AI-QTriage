"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { api, getApiUrl } from "@/lib/api";
import { captureBrowserLocation, type SosGeoPayload } from "@/lib/geo";
import { 
  ShieldAlert, 
  Activity, 
  AlertTriangle, 
  Cpu, 
  Layers,
  ArrowLeft,
  RefreshCw,
  Sliders,
  CheckCircle2,
  FileText
} from "lucide-react";

import { ReviewStatusBanner } from "./components/ReviewStatusBanner";
import { ImageAnalysis } from "./components/ImageAnalysis";
import { QuestionnaireSummary } from "./components/QuestionnaireSummary";
import { SensorSummary } from "./components/SensorSummary";
import { MultimodalResults } from "./components/MultimodalResults";
import { SafetyGuidance } from "./components/SafetyGuidance";
import { TechnicalAudit } from "./components/TechnicalAudit";

export default function CaseDetails() {
  const params = useParams();
  const router = useRouter();
  const caseId = params.id as string;

  const [caseData, setCaseData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [liveYoloClasses, setLiveYoloClasses] = useState<string[]>([]);

  // Two primary views: "overview" (default) vs "technical"
  const [viewMode, setViewMode] = useState<"overview" | "technical">("overview");

  // Verifiable test run logs state
  const [testRunData, setTestRunData] = useState<any>(null);
  
  // SOS countdown status state
  const [sosStatus, setSosStatus] = useState<any>(null);
  const [sosMode, setSosMode] = useState<"local_demo" | "twilio_test">("local_demo");
  const [twilioConfig, setTwilioConfig] = useState<any>(null);
  const [sosGeo, setSosGeo] = useState<SosGeoPayload | null>(null);
  const [sosGeoStatus, setSosGeoStatus] = useState<"idle" | "pending" | "ready" | "denied">("idle");
  const [manualLat, setManualLat] = useState("");
  const [manualLng, setManualLng] = useState("");
  const [sosError, setSosError] = useState<string | null>(null);

  useEffect(() => {
    async function loadCase() {
      if (!caseId) return;
      try {
        const [data, twCfg, models, vRun] = await Promise.all([
          api.getCase(caseId),
          api.getSOSConfig(),
          api.getModels().catch(() => []),
          api.getVerificationRun(caseId).catch(() => null),
        ]);
        setCaseData(data);
        setTwilioConfig(twCfg);
        if (vRun) setTestRunData(vRun);
        if (data?.sos_user_location?.latitude != null && data?.sos_user_location?.longitude != null) {
          setSosGeo(data.sos_user_location as any);
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
      console.error("Failed to respond to SOS:", err);
      setSosError(err?.message || "Failed to record response.");
    }
  };

  // Loading Screen
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-4">
        <RefreshCw className="h-8 w-8 text-blue-500 animate-spin" />
        <p className="text-sm font-mono text-slate-400">Loading verified case evidence...</p>
      </div>
    );
  }

  // Error Screen
  if (error || !caseData) {
    return (
      <div className="max-w-2xl mx-auto my-12 p-8 bg-slate-900 border border-slate-800 rounded-3xl text-center space-y-4">
        <div className="w-12 h-12 bg-red-950/40 text-red-400 rounded-2xl flex items-center justify-center mx-auto border border-red-900/50">
          <AlertTriangle className="h-6 w-6" />
        </div>
        <h2 className="text-xl font-bold text-white">Error Retrieving Case</h2>
        <p className="text-sm text-slate-400 font-mono">{error || "Case record not found in research database."}</p>
        <Link
          href="/"
          className="inline-flex items-center gap-2 px-5 py-2.5 bg-slate-800 hover:bg-slate-700 text-white text-xs font-bold rounded-xl transition-all"
        >
          <ArrowLeft className="h-4 w-4" /> Return to Cases
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-16 px-4 sm:px-6">
      {/* Top Navigation & View Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-2">
        <Link
          href="/"
          className="inline-flex items-center gap-1.5 text-xs font-bold text-slate-400 hover:text-white transition-colors self-start"
        >
          <ArrowLeft className="h-4 w-4" /> Back to Cases
        </Link>

        {/* Primary View Switcher: Case Overview vs Technical Details & Audit */}
        <div className="flex items-center gap-1 bg-slate-900 p-1.5 rounded-2xl border border-slate-800 shadow-lg self-start sm:self-auto">
          <button
            type="button"
            onClick={() => setViewMode("overview")}
            className={`px-4 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-2 ${
              viewMode === "overview"
                ? "bg-blue-600 text-white shadow-md shadow-blue-600/30"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
            }`}
          >
            <Activity className="h-3.5 w-3.5" />
            <span>Case Overview</span>
          </button>
          <button
            type="button"
            onClick={() => setViewMode("technical")}
            className={`px-4 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-2 ${
              viewMode === "technical"
                ? "bg-purple-600 text-white shadow-md shadow-purple-600/30"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
            }`}
          >
            <Cpu className="h-3.5 w-3.5" />
            <span>Technical Details &amp; Audit</span>
          </button>
        </div>
      </div>

      {/* VIEW 1: CASE OVERVIEW (The 6 Compact Sections) */}
      {viewMode === "overview" && (
        <div className="space-y-8 animate-fadeIn">
          {/* SECTION 1: Case Overview & Review Status */}
          <ReviewStatusBanner caseData={caseData} caseId={caseId} />

          {/* SECTION 2: Image Analysis (YOLO11, EfficientNetV2, ResNet34-U-Net + Disagreement Alert + Visual Artifacts) */}
          <ImageAnalysis caseData={caseData} liveYoloClasses={liveYoloClasses} />

          {/* SECTION 3: Questionnaire Context */}
          <QuestionnaireSummary caseData={caseData} />

          {/* SECTION 4: Sensor Analysis (4 Stat Cards + Interpretation Note) */}
          <SensorSummary caseData={caseData} />

          {/* SECTION 5: Multimodal AI Results & Explainability (XGBoost, VQC Excluded, Compact SHAP) */}
          <MultimodalResults caseData={caseData} />

          {/* SECTION 6: Safety Guidance & SOS Status */}
          <SafetyGuidance caseData={caseData} sosStatus={sosStatus} />

          {/* Bottom Switcher Card to Explore Technical Details */}
          <div className="p-5 rounded-2xl bg-gradient-to-r from-slate-900 via-purple-950/20 to-slate-900 border border-purple-900/40 flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-xl">
            <div className="space-y-1">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Cpu className="h-4 w-4 text-purple-400" />
                Inspect Model Architectures, Quantum Benchmarks &amp; Verification Proof
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Review complete 23-feature SHAP breakdowns, confusion matrices, counterfactual sweeps, and multi-suite verification test logs.
              </p>
            </div>
            <button
              type="button"
              onClick={() => {
                setViewMode("technical");
                if (typeof window !== "undefined") window.scrollTo({ top: 0, behavior: "smooth" });
              }}
              className="px-4 py-2.5 bg-purple-600 hover:bg-purple-500 text-white rounded-xl text-xs font-bold transition-all shadow-md shrink-0 flex items-center gap-2"
            >
              <span>Switch to Technical Details &amp; Audit</span>
              <span>&rarr;</span>
            </button>
          </div>
        </div>
      )}

      {/* VIEW 2: TECHNICAL DETAILS & AUDIT (Preserving Full Research & Verification Evidence) */}
      {viewMode === "technical" && (
        <div className="space-y-6 animate-fadeIn">
          {/* Quick Back to Overview Link */}
          <div className="flex items-center justify-between p-4 bg-slate-900/60 rounded-xl border border-slate-800 text-xs">
            <span className="text-slate-400">
              Viewing Full Technical Diagnostics &amp; Audit Workspace
            </span>
            <button
              type="button"
              onClick={() => {
                setViewMode("overview");
                if (typeof window !== "undefined") window.scrollTo({ top: 0, behavior: "smooth" });
              }}
              className="text-cyan-400 hover:text-cyan-300 font-bold underline flex items-center gap-1"
            >
              &larr; Return to Case Overview
            </button>
          </div>

          <TechnicalAudit caseData={caseData} testRunData={testRunData} />
        </div>
      )}
    </div>
  );
}
