"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { api, getApiUrl } from "@/lib/api";
import ProgressStepper from "@/components/ProgressStepper";
import {
  ArrowLeft,
  ArrowRight,
  Upload,
  Activity,
  Loader2,
  AlertCircle,
  Zap,
  Sliders,
  ShieldAlert,
  Smartphone
} from "lucide-react";
import { RealTimeSensorCapture } from "@/components/RealTimeSensorCapture";

export default function CreateCase() {
  const router = useRouter();
  const [isMounted, setIsMounted] = useState(false);
  const [step, setStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setIsMounted(true);
  }, []);

  // Case state
  const [caseId, setCaseId] = useState<string | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [caseObj, setCaseObj] = useState<any>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  // Questionnaire template state
  const [templateData, setTemplateData] = useState<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [templateLoading, setTemplateLoading] = useState(false);
  
  // Canonical Questionnaire State: All fields default to "not_provided" unless explicitly answered
  const [answers, setAnswers] = useState<Record<string, any>>({
    pain_level: "",
    location: "not_provided",
    cause: "not_provided",
    onset_hours: "not_provided",
    movement_limitation: "not_provided",
    weight_bearing: "not_provided",
    swelling: "not_provided",
    bruising_discoloration: "not_provided",
    redness: "not_provided",
    warmth: "not_provided",
    open_wound: "not_provided",
    bleeding: "not_provided",
    crack_pop: "not_provided",
    deformity: "not_provided",
    numbness_tingling: "not_provided",
    previous_injury: "not_provided",
    symptom_progression: "not_provided"
  });

  // Sensor state
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [sensorMode, setSensorMode] = useState<"none" | "demo" | "upload" | "simulate" | "live">("none");
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [sensorFile, setSensorFile] = useState<File | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [sensorUploaded, setSensorUploaded] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [sensorSkipped, setSensorSkipped] = useState(false);
  const [sensorStatusMessage, setSensorStatusMessage] = useState<string | null>(null);
  const [showLiveSensorModal, setShowLiveSensorModal] = useState(false);
  const [showSimulateModal, setShowSimulateModal] = useState(false);

  const handleSimulateSensor = async (scenario: string) => {
    if (!caseId) return;
    setLoading(true);
    setError(null);
    try {
      await api.simulateSensor(caseId, scenario);
      setSensorUploaded(true);
      setSensorMode("simulate");
      setSensorStatusMessage(`✓ Simulated sensor log generated (${scenario})`);
      setShowSimulateModal(false);
      setStep(5);
    } catch (err: any) {
      setError(err.message || "Failed to simulate sensor data.");
    } finally {
      setLoading(false);
    }
  };

  // ── Step 1: Initialize Case ──────────────────────────────────────────────────
  const startCase = async () => {
    setLoading(true);
    setError(null);
    try {
      const newCase = await api.createCase("New Assessment Initial Instance");
      setCaseId(newCase.case_id);
      setCaseObj(newCase);
      setStep(2);
    } catch (err: any) {
      setError(err.message || "Failed to initialize case in database.");
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  // ── Step 2: Upload Image ─────────────────────────────────────────────────────
  const handleImageUpload = async () => {
    if (!imageFile || !caseId) return;
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", imageFile);
    try {
      const response = await fetch(getApiUrl(`/api/cases/${caseId}/image`), {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const errorText = await response.text();
        let msg = "Failed to validate image quality.";
        try { msg = JSON.parse(errorText).detail || msg; } catch {}
        throw new Error(msg);
      }
      // Vision analysis first so questionnaire routing can use visible_injury.
      // First analyze on CPU can take several minutes while models load.
      const analyzeController = new AbortController();
      const analyzeTimer = setTimeout(() => analyzeController.abort(), 600_000);
      let analyzeRes: Response;
      try {
        analyzeRes = await fetch(getApiUrl(`/api/cases/${caseId}/analyze`), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: analyzeController.signal,
        });
      } finally {
        clearTimeout(analyzeTimer);
      }
      if (!analyzeRes.ok) {
        const detail = await analyzeRes.text();
        throw new Error(detail || "Image analysis failed before questionnaire routing.");
      }
      await fetchQuestionnaireTemplate();
      const updated = await api.getCase(caseId);
      setCaseObj(updated);
      setStep(3);
    } catch (err: any) {
      setError(err.message || "Error uploading image.");
    } finally {
      setLoading(false);
    }
  };

  const fetchQuestionnaireTemplate = async () => {
    if (!caseId) return;
    setTemplateLoading(true);
    try {
      const delays = [0, 1500, 3000];
      for (let attempt = 0; attempt < delays.length; attempt++) {
        if (delays[attempt] > 0) {
          await new Promise((resolve) => setTimeout(resolve, delays[attempt]));
        }
        try {
          const result = await api.getQuestionnaireTemplate(caseId);
          setTemplateData(result);
          return;
        } catch (err) {
          if (attempt === delays.length - 1) {
            console.warn("Questionnaire template unavailable after retries:", err);
            setTemplateData(null);
          }
        }
      }
    } finally {
      setTemplateLoading(false);
    }
  };

  // ── Step 3: Manual Questionnaire ─────────────────────────────────────────────
  const handleAnswerChange = (questionId: string, value: any) => {
    setAnswers((prev) => ({ ...prev, [questionId]: value }));
  };

  const handleQuestionnaireSubmit = async () => {
    if (!caseId) return;
    setLoading(true);
    setError(null);
    const payload = {
      answers,
      template_id: templateData?.routed && templateData?.template_id ? templateData.template_id : "generic_research_v1",
      template_version: templateData?.template?.template_version || "1.0",
      answer_source: "typed",
    };
    try {
      await api.submitQuestionnaire(caseId, payload);
      const updated = await api.getCase(caseId);
      setCaseObj(updated);
      setStep(4);
    } catch (err: any) {
      setError(err.message || "Error submitting questionnaire.");
    } finally {
      setLoading(false);
    }
  };

  // ── Step 4: Sensor Options ───────────────────────────────────────────────────
  const handleSkipSensor = async () => {
    if (!caseId) return;
    setLoading(true);
    setError(null);
    try {
      await api.skipSensor(caseId);
      setSensorSkipped(true);
      setSensorMode("none");
      setSensorStatusMessage("Sensor data omitted. Continuing with reduced-modality pipeline.");
      setStep(5);
    } catch (err: any) {
      setError(err.message || "Failed to record sensor skip.");
    } finally {
      setLoading(false);
    }
  };

  const handleUseDemo = async () => {
    if (!caseId) return;
    setLoading(true);
    setError(null);
    try {
      await api.loadDemoSensor(caseId);
      setSensorUploaded(true);
      setSensorMode("demo");
      setSensorStatusMessage("✓ Demo sensor log loaded (Football Fall Data)");
      setStep(5);
    } catch (err: any) {
      setError(err.message || "Failed to load demo sensor log.");
    } finally {
      setLoading(false);
    }
  };

  const handleSensorUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !caseId) return;
    setSensorFile(file);
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(getApiUrl(`/api/cases/${caseId}/sensor`), {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("Sensor file upload failed.");
      setSensorUploaded(true);
      setSensorMode("upload");
      setSensorStatusMessage("✓ Valid sensor file uploaded");
      setStep(5);
    } catch (err: any) {
      setError(err.message || "Error uploading sensor file.");
    } finally {
      setLoading(false);
    }
  };

  // ── Step 5: Run Multimodal Model Execution ───────────────────────────────────
  const runAnalysis = async () => {
    if (!caseId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(getApiUrl(`/api/cases/${caseId}/analyze`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!res.ok) throw new Error("Model execution failed.");
      router.push(`/cases/${caseId}`);
    } catch (err: any) {
      setError(err.message || "Analysis failed.");
      setLoading(false);
    }
  };

  const QUESTION_FIELDS = [
    { id: "location", question: "1. Injury Location", type: "text", placeholder: "e.g. Left ankle, Right knee (Leave blank for Not provided)" },
    { id: "pain_level", question: "2. Pain Level (0 - 10)", type: "select", options: [
        { value: "not_provided", label: "Not provided" },
        { value: "0", label: "0 - No pain" },
        { value: "1", label: "1 - Mild pain" },
        { value: "2", label: "2 - Mild pain" },
        { value: "3", label: "3 - Moderate pain" },
        { value: "4", label: "4 - Moderate pain" },
        { value: "5", label: "5 - Moderate pain" },
        { value: "6", label: "6 - Severe pain" },
        { value: "7", label: "7 - Severe pain" },
        { value: "8", label: "8 - Very severe pain" },
        { value: "9", label: "9 - Very severe pain" },
        { value: "10", label: "10 - Worst pain possible" }
      ]
    },
    { id: "cause", question: "3. Cause / Injury Mechanism", type: "select", options: [
        { value: "not_provided", label: "Not provided" },
        { value: "fall", label: "Fall" },
        { value: "sports", label: "Sports Injury" },
        { value: "twist", label: "Twist / Sprain" },
        { value: "direct_blow", label: "Direct Impact / Blow" },
        { value: "overuse", label: "Overuse / Repetitive" },
        { value: "other", label: "Other" }
      ]
    },
    { id: "onset_hours", question: "4. When did the injury occur?", type: "select", options: [
        { value: "not_provided", label: "Not provided" },
        { value: "< 1 hour ago", label: "< 1 hour ago" },
        { value: "1-3 hours ago", label: "1 - 3 hours ago" },
        { value: "3-12 hours ago", label: "3 - 12 hours ago" },
        { value: "12-24 hours ago", label: "12 - 24 hours ago" },
        { value: "> 24 hours ago", label: "> 24 hours ago" }
      ]
    },
    { id: "movement_limitation", question: "5. Movement Limitation", type: "select", options: [
        { value: "not_provided", label: "Not provided" },
        { value: "none", label: "No movement limitation" },
        { value: "mild", label: "Mild discomfort on movement" },
        { value: "moderate", label: "Moderate pain on movement" },
        { value: "severe", label: "Severe pain / unable to move" }
      ]
    },
    { id: "weight_bearing", question: "6. Weight-Bearing Ability", type: "select", options: [
        { value: "not_provided", label: "Not provided" },
        { value: "full", label: "Full weight-bearing without pain" },
        { value: "partial", label: "Partial weight-bearing with limp/pain" },
        { value: "unable", label: "Unable to bear weight" }
      ]
    },
    { id: "swelling", question: "7. Visible Swelling Present?", type: "select", options: [
        { value: "not_provided", label: "Not provided" },
        { value: "yes", label: "Yes" },
        { value: "no", label: "No" },
        { value: "unsure", label: "Unsure" }
      ]
    },
    { id: "bruising_discoloration", question: "8. Bruising or Discoloration?", type: "select", options: [
        { value: "not_provided", label: "Not provided" },
        { value: "yes", label: "Yes" },
        { value: "no", label: "No" }
      ]
    },
    { id: "redness", question: "9. Skin Redness Present?", type: "select", options: [
        { value: "not_provided", label: "Not provided" },
        { value: "yes", label: "Yes" },
        { value: "no", label: "No" }
      ]
    },
    { id: "warmth", question: "10. Area Warm to Touch?", type: "select", options: [
        { value: "not_provided", label: "Not provided" },
        { value: "yes", label: "Yes" },
        { value: "no", label: "No" }
      ]
    },
    { id: "open_wound", question: "11. Open Wound or Skin Cut?", type: "select", options: [
        { value: "not_provided", label: "Not provided" },
        { value: "yes", label: "Yes" },
        { value: "no", label: "No" }
      ]
    },
    { id: "bleeding", question: "12. Active Bleeding?", type: "select", options: [
        { value: "not_provided", label: "Not provided" },
        { value: "none", label: "No bleeding" },
        { value: "minor", label: "Minor oozing / scrape" },
        { value: "active", label: "Active bleeding" }
      ]
    },
    { id: "crack_pop", question: "13. Crack or Popping Sound Heard at Impact?", type: "select", options: [
        { value: "not_provided", label: "Not provided" },
        { value: "yes", label: "Yes" },
        { value: "no", label: "No" },
        { value: "unsure", label: "Unsure" }
      ]
    },
    { id: "deformity", question: "14. Visible Deformity or Abnormal Shape?", type: "select", options: [
        { value: "not_provided", label: "Not provided" },
        { value: "yes", label: "Yes" },
        { value: "no", label: "No" }
      ]
    },
    { id: "numbness_tingling", question: "15. Numbness or Tingling Sensation?", type: "select", options: [
        { value: "not_provided", label: "Not provided" },
        { value: "yes", label: "Yes" },
        { value: "no", label: "No" }
      ]
    },
    { id: "previous_injury", question: "16. Previous Injury to Same Area?", type: "select", options: [
        { value: "not_provided", label: "Not provided" },
        { value: "yes", label: "Yes" },
        { value: "no", label: "No" }
      ]
    },
    { id: "symptom_progression", question: "17. Symptom Progression", type: "select", options: [
        { value: "not_provided", label: "Not provided" },
        { value: "improving", label: "Improving" },
        { value: "stable", label: "Stable / Unchanged" },
        { value: "worsening", label: "Worsening / Rapidly swelling" }
      ]
    }
  ];

  const renderQuestion = (q: any) => {
    const val = answers[q.id] ?? "not_provided";
    if (q.options && q.options.length > 0) {
      return (
        <select
          key={q.id}
          value={val}
          onChange={(e) => handleAnswerChange(q.id, e.target.value)}
          className="w-full p-2.5 bg-[#080D1C] border border-[#26324A] rounded-xl text-xs text-slate-200 focus:border-blue-500 focus:outline-none"
        >
          {q.options.map((opt: any) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      );
    }
    return (
      <input
        key={q.id}
        type="text"
        value={val === "not_provided" ? "" : val}
        placeholder={q.placeholder || ""}
        onChange={(e) => handleAnswerChange(q.id, e.target.value)}
        className="w-full p-2.5 bg-[#080D1C] border border-[#26324A] rounded-xl text-xs text-slate-200 focus:border-blue-500 focus:outline-none"
      />
    );
  };

  if (!isMounted) {
    return (
      <div className="p-8 text-center text-slate-400 text-xs flex items-center justify-center space-x-2">
        <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
        <span>Loading New Assessment workspace...</span>
      </div>
    );
  }

  return (
    <div className="space-y-4 sm:space-y-6 max-w-4xl mx-auto py-2 sm:py-4 px-0">
      {/* 1. TOP PROGRESS STEPPER */}
      <ProgressStepper currentStep={step} />

      {/* Global Error Notice */}
      {error && (
        <div className="p-4 bg-red-950/30 border border-red-800/60 rounded-2xl flex items-center space-x-3 text-red-300 text-xs">
          <AlertCircle className="h-5 w-5 text-red-400 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* STEP 1: INITIALIZE */}
      {step === 1 && (
        <div className="dash-card p-5 sm:p-8 text-center space-y-6 max-w-xl mx-auto my-4 sm:my-8 border border-[#26324A] bg-[#0B1224] rounded-2xl shadow-xl">
          <div className="h-16 w-16 bg-blue-600/20 border border-blue-500/40 text-blue-400 rounded-2xl flex items-center justify-center mx-auto glow-blue">
            <Activity className="h-8 w-8 animate-pulse" />
          </div>
          <div className="space-y-2">
            <h3 className="text-xl font-extrabold text-white">Initialize Research Session</h3>
            <p className="text-xs text-slate-400">
              Generates a unique MongoDB case ID for tracking vision metrics, questionnaires, and sensor data.
            </p>
          </div>
          <button
            onClick={startCase}
            disabled={loading}
            className="px-6 py-3 bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs rounded-xl flex items-center space-x-2 mx-auto disabled:opacity-50 transition-all glow-blue"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
            <span>Initialize Research Session</span>
            <ArrowRight className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* STEP 2: IMAGE UPLOAD */}
      {step === 2 && (
        <div className="dash-card p-6 space-y-6 max-w-2xl mx-auto border border-[#26324A] bg-[#0B1224] rounded-2xl shadow-xl">
          <div className="space-y-1 text-center">
            <h3 className="text-lg font-bold text-white">Upload Injury Photograph</h3>
            <p className="text-xs text-slate-400">
              Accepts rectangular photographs (e.g. 1024×768) with automatic aspect-ratio letterboxing.
            </p>
            {caseId && <p className="text-[11px] font-mono text-emerald-400">Active Case ID: {caseId}</p>}
          </div>

          <div className="border-2 border-dashed border-[#26324A] hover:border-blue-500/50 rounded-2xl p-8 text-center bg-[#080D1C]/50 transition-all">
            {imagePreview ? (
              <div className="space-y-4">
                <img src={imagePreview} alt="Injury Preview" className="max-h-64 mx-auto rounded-xl object-contain border border-[#26324A]" />
                <button onClick={() => { setImageFile(null); setImagePreview(null); }} className="text-xs text-red-400 hover:underline">
                  Remove Photo
                </button>
              </div>
            ) : (
              <label className="cursor-pointer block space-y-4">
                <Upload className="h-12 w-12 text-slate-500 mx-auto" />
                <span className="block text-xs font-semibold text-slate-300">Click or drag photo here</span>
                <span className="block text-[10px] text-slate-500">JPG, JPEG, PNG format</span>
                <input type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
              </label>
            )}
          </div>

          <div className="flex justify-between items-center pt-2">
            <button onClick={() => setStep(1)} className="text-xs text-slate-400 hover:text-slate-200 flex items-center space-x-1">
              <ArrowLeft className="h-3 w-3" /><span>Back</span>
            </button>
            <button
              onClick={handleImageUpload}
              disabled={!imageFile || loading}
              className="px-6 py-2.5 bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs rounded-xl flex items-center space-x-2 disabled:opacity-50 transition-all"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              <span>Validate &amp; Analyze Image</span>
              <ArrowRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      {/* STEP 3: QUESTIONNAIRE */}
      {step === 3 && (
        <div className="dash-card p-6 space-y-6 max-w-2xl mx-auto border border-[#26324A] bg-[#0B1224] rounded-2xl shadow-xl">
          <div className="flex justify-between items-center border-b border-[#26324A] pb-3">
            <div>
              <h3 className="text-base font-bold text-white">Injury-Specific Questionnaire</h3>
              <p className="text-xs text-slate-400">
                Routed Template: <strong className="text-emerald-400">{templateData?.routed ? templateData.template_id : "generic_research_v1"}</strong>
                {templateData?.routed ? " (from vision analysis)" : " (generic until a class-specific template is available)"}
              </p>
              <p className="text-[11px] text-slate-500 mt-1">
                Form fields are the canonical 23-feature fusion schema. Per-class template JSON files use incompatible IDs (e.g. pain vs pain_level) and are routing metadata only.
              </p>
            </div>
          </div>

          <div className="space-y-4 text-xs max-h-[480px] overflow-y-auto pr-2">
            {QUESTION_FIELDS.map((q: any) => (
              <div key={q.id} className="space-y-1.5">
                <label className="font-semibold text-slate-300 block">{q.question}</label>
                {renderQuestion(q)}
              </div>
            ))}
          </div>

          <div className="flex justify-between items-center pt-4 border-t border-[#26324A]">
            <button onClick={() => setStep(2)} className="text-xs text-slate-400 hover:text-slate-200 flex items-center space-x-1">
              <ArrowLeft className="h-3 w-3" /><span>Back</span>
            </button>
            <button
              onClick={handleQuestionnaireSubmit}
              disabled={loading}
              className="px-6 py-2.5 bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs rounded-xl flex items-center space-x-2 disabled:opacity-50 transition-all"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              <span>Save &amp; Continue to Sensor Step</span>
              <ArrowRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      {/* STEP 4: SENSOR DATA OPTIONS */}
      {step === 4 && (
        <div className="dash-card p-6 space-y-6 max-w-3xl mx-auto border border-[#26324A] bg-[#0B1224] rounded-2xl shadow-xl">
          <div className="space-y-1 text-center">
            <h3 className="text-lg font-bold text-white">Smartphone Sensor Data (Optional)</h3>
            <p className="text-xs text-slate-400">
              Sensor data provides peak impact force and physical stabilization metrics. Choose from real-time device capture, log upload, demo datasets, simulation, or skip.
            </p>
          </div>

          {showLiveSensorModal && caseId ? (
            <RealTimeSensorCapture
              caseId={caseId}
              onSuccess={() => {
                setSensorStatusMessage("✓ Real-time device sensor data captured and processed");
                setShowLiveSensorModal(false);
                setStep(5);
              }}
              onCancel={() => setShowLiveSensorModal(false)}
            />
          ) : showSimulateModal ? (
            <div className="p-6 bg-[#0D1426] border border-[#26324A] rounded-2xl space-y-4 max-w-lg mx-auto text-xs">
              <div className="flex justify-between items-center border-b border-slate-800 pb-2">
                <span className="font-bold text-white text-sm">Select Simulation Scenario</span>
                <button onClick={() => setShowSimulateModal(false)} className="text-slate-400 hover:text-white">✕</button>
              </div>
              <p className="text-slate-400">Choose a physical kinematic scenario to simulate sensor signals:</p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <button onClick={() => handleSimulateSensor("football_fall")} className="p-3 bg-slate-900 border border-slate-800 hover:border-blue-500 rounded-xl text-left space-y-1">
                  <strong className="block text-blue-400">Football Fall</strong>
                  <span className="text-[10px] text-slate-400">High peak G-force with rotation</span>
                </button>
                <button onClick={() => handleSimulateSensor("sudden_fall")} className="p-3 bg-slate-900 border border-slate-800 hover:border-blue-500 rounded-xl text-left space-y-1">
                  <strong className="block text-purple-400">Sudden Fall</strong>
                  <span className="text-[10px] text-slate-400">Freefall phase followed by impact</span>
                </button>
                <button onClick={() => handleSimulateSensor("sudden_impact")} className="p-3 bg-slate-900 border border-slate-800 hover:border-blue-500 rounded-xl text-left space-y-1">
                  <strong className="block text-red-400">Sudden Impact</strong>
                  <span className="text-[10px] text-slate-400">Severe linear kinetic deceleration</span>
                </button>
                <button onClick={() => handleSimulateSensor("normal_movement")} className="p-3 bg-slate-900 border border-slate-800 hover:border-blue-500 rounded-xl text-left space-y-1">
                  <strong className="block text-emerald-400">Normal Movement</strong>
                  <span className="text-[10px] text-slate-400">Baseline walking telemetry</span>
                </button>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
              {/* 1. NEW: Capture Real-Time Sensor Data */}
              <button
                type="button"
                onClick={() => setShowLiveSensorModal(true)}
                className="p-4 rounded-xl border-2 border-emerald-500/80 bg-emerald-950/20 hover:bg-emerald-950/40 text-left space-y-2 transition-all col-span-1 sm:col-span-2 lg:col-span-1 shadow-lg shadow-emerald-950/50 glow-blue"
              >
                <div className="flex items-center space-x-2 text-emerald-400">
                  <Smartphone className="h-6 w-6" />
                  <span className="px-1.5 py-0.5 bg-emerald-900 text-emerald-300 text-[9px] font-bold rounded">NEW</span>
                </div>
                <div>
                  <span className="font-extrabold block text-xs text-emerald-300">📱 Capture Real-Time Sensor Data</span>
                  <p className="text-[10px] text-slate-400 mt-1">Record live motion hardware from your mobile browser</p>
                </div>
              </button>

              {/* 2. Upload CSV/JSON */}
              <label className="p-4 rounded-xl border border-[#26324A] bg-[#0D1426] hover:border-blue-500 cursor-pointer text-left space-y-2 transition-all block">
                <Upload className="h-6 w-6 text-purple-400" />
                <div>
                  <span className="font-bold block text-xs text-slate-200">📁 Upload Sensor Log</span>
                  <p className="text-[10px] text-slate-400 mt-1">Select .csv or .json log file</p>
                </div>
                <input type="file" accept=".csv,.json" className="hidden" onChange={handleSensorUpload} />
              </label>

              {/* 3. Demo Sensor */}
              <button
                type="button"
                onClick={handleUseDemo}
                className="p-4 rounded-xl border border-[#26324A] bg-[#0D1426] hover:border-blue-500 text-left space-y-2 transition-all"
              >
                <Zap className="h-6 w-6 text-blue-400" />
                <div>
                  <span className="font-bold block text-xs text-slate-200">🧪 Use Demo Log</span>
                  <p className="text-[10px] text-slate-400 mt-1">Load sample football fall dataset</p>
                </div>
              </button>

              {/* 4. Simulate Sensor Data */}
              <button
                type="button"
                onClick={() => setShowSimulateModal(true)}
                className="p-4 rounded-xl border border-[#26324A] bg-[#0D1426] hover:border-blue-500 text-left space-y-2 transition-all"
              >
                <Sliders className="h-6 w-6 text-indigo-400" />
                <div>
                  <span className="font-bold block text-xs text-slate-200">⚙ Simulate Sensor Data</span>
                  <p className="text-[10px] text-slate-400 mt-1">Generate kinematic scenario vectors</p>
                </div>
              </button>

              {/* 5. Continue Without Sensor */}
              <button
                type="button"
                onClick={handleSkipSensor}
                className="p-4 rounded-xl border border-amber-800/60 bg-amber-950/20 hover:bg-amber-950/40 text-left space-y-2 transition-all"
              >
                <ShieldAlert className="h-6 w-6 text-amber-400" />
                <div>
                  <span className="font-bold block text-xs text-amber-300">⊘ Continue Without Sensor</span>
                  <p className="text-[10px] text-slate-400 mt-1">Omit sensor logs and proceed</p>
                </div>
              </button>
            </div>
          )}

          {sensorStatusMessage && !showLiveSensorModal && !showSimulateModal && (
            <div className="p-3.5 bg-[#0D1426] border border-[#26324A] rounded-xl text-xs text-emerald-400 font-semibold text-center">
              {sensorStatusMessage}
            </div>
          )}

          <div className="flex justify-between items-center pt-2 border-t border-[#26324A]">
            <button onClick={() => setStep(3)} className="text-xs text-slate-400 hover:text-slate-200 flex items-center space-x-1">
              <ArrowLeft className="h-3 w-3" /><span>Back to Questionnaire</span>
            </button>
          </div>
        </div>
      )}

      {/* STEP 5: RUN MULTIMODAL MODEL ANALYSIS */}
      {step === 5 && (
        <div className="dash-card p-8 text-center space-y-6 max-w-xl mx-auto border border-[#26324A] bg-[#0B1224] rounded-2xl shadow-xl my-8">
          <div className="h-16 w-16 bg-purple-600/20 border border-purple-500/40 text-purple-400 rounded-2xl flex items-center justify-center mx-auto glow-purple">
            <Zap className="h-8 w-8 animate-pulse" />
          </div>
          <div className="space-y-2">
            <h3 className="text-xl font-extrabold text-white">Execute Multimodal AI Pipeline</h3>
            <p className="text-xs text-slate-400">
              Executes YOLO11, EfficientNetV2, U-Net, Grad-CAM, XGBoost + SHAP, PCA, VQC quantum classifier, rules engine, and report generation.
            </p>
            {sensorStatusMessage && (
              <p className="text-xs text-emerald-400 font-semibold pt-1">{sensorStatusMessage}</p>
            )}
          </div>

          <button
            type="button"
            onClick={runAnalysis}
            disabled={loading}
            className="px-8 py-3.5 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white font-bold text-xs rounded-xl flex items-center space-x-2 mx-auto disabled:opacity-50 transition-all shadow-xl shadow-purple-600/30"
          >
            {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Zap className="h-5 w-5" />}
            <span>Run Full Multimodal Analysis</span>
            <ArrowRight className="h-5 w-5" />
          </button>
        </div>
      )}
    </div>
  );
}
