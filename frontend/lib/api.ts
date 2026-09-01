export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export function getApiUrl(path: string): string {
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL}${cleanPath}`;
}

export function getUploadUrl(pathOrFilename: string): string {
  if (!pathOrFilename) return "";
  if (pathOrFilename.startsWith("http://") || pathOrFilename.startsWith("https://")) {
    return pathOrFilename;
  }
  const filename = pathOrFilename.split("/").pop() || pathOrFilename;
  return `${API_BASE_URL}/uploads/${filename}`;
}

export interface Case {
  case_id: string;
  created_at: string;
  status: string;
  image_reference?: string | null;
  data_provenance?: string;
  research_demo_warning?: string | null;
  prediction_agreement?: string | null;
  uncertainty_level?: string | null;
  uncertainty_reasons?: string[] | null;
  visible_injury?: {
    finding: string;
    finding_detected?: boolean;
    detection_message?: string;
    yolo_finding?: string | null;
    yolo_finding_detected?: boolean;
    yolo_confidence?: number | null;
    yolo_bounding_box?: number[] | null;
    yolo_supported_classes?: string[];
    classifier_finding?: string | null;
    classifier_probability?: number | null;
    confidence: number | null;
    bounding_box: number[] | null;
    bounding_box_model?: number[] | null;
    affected_ratio?: number | null;
    affected_area_ratio?: number | null;
    segmentation_available?: boolean;
    segmentation_reliable?: boolean;
    segmentation_reason?: string;
    classification?: Record<string, number>;
    overlay_url?: string;
    overlay_width?: number;
    overlay_height?: number;
    mask_url?: string;
    image_url?: string;
    original_width?: number;
    original_height?: number;
    segmentation_status?: string;
    segmentation_model_status?: string;
    segmentation_message?: string;
    segmentation_trust?: string;
    denominator_label?: string;
    source_type?: string;
    data_provenance?: string;
    display_message?: string;
    gradcam_label?: string;
    gradcam_explanation?: string;
    gradcam_reliability?: string;
    gradcam_source_model?: string;
    gradcam_predicted_class?: string | null;
    gradcam_confidence?: number | null;
    gradcam_model_status?: string;
    gradcam_explanation_status?: string;
    gradcam_overlay_generated?: boolean;
    gradcam_withheld_reason?: string | null;
                    classifier_status?: string;
    classifier_model_status?: string;
    classifier_reason?: string;
    classifier_is_confident?: boolean;
  } | null;
  questionnaire?: {
    answers: Record<string, any>;
    voice_used?: boolean;
    voice_transcript?: string;
    template_id?: string;
    template_version?: string;
    answer_source?: string;
  } | null;
  sensor_summary?: Record<string, any> | null;
  sensor_available?: boolean;
  sensor_source_type?: string;
  xgboost_prediction?: {
    class: string;
    probability: number;
    model_version?: string;
    model_id?: string;
    n_features?: number;
    data_provenance?: string;
    data_provenance_detail?: string;
    status?: string;
    artifact_path?: string;
  } | null;
  quantum_prediction?: {
    class: string | null;
    score?: number[] | null;
    model_version?: string;
    model_id?: string;
    status?: string;
    experimental?: boolean;
    experimental_only?: boolean;
    used_in_main_decision?: boolean;
    data_provenance?: string;
    error?: string | null;
  } | null;
  agreement_score?: string | null;
  uncertainty_status?: string | null;
  safety_information?: string[] | null;
  sos_status?: string | null;
  sos_event_id?: string | null;
  report_reference?: string | null;
  is_demo?: boolean;
  modalities_used?: string[];
  model_configuration_used?: string | null;
}

export async function apiFetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const headers: Record<string, string> = {
      ...(options?.headers as Record<string, string> || {}),
    };

    if (!(options?.body instanceof FormData) && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }

    const response = await fetch(url, {
      ...options,
      headers,
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      let errorMessage = "An error occurred with the request.";
      try {
        const errorJson = JSON.parse(errorText);
        const detail = errorJson.detail ?? errorJson.message ?? errorJson.error;
        if (typeof detail === "string") {
          errorMessage = detail;
        } else if (Array.isArray(detail)) {
          errorMessage = detail
            .map((d) => (typeof d === "string" ? d : d?.msg || JSON.stringify(d)))
            .join("; ");
        } else if (detail != null) {
          errorMessage = String(detail);
        }
      } catch {
        errorMessage = errorText || errorMessage;
      }
      throw new Error(errorMessage);
    }
    
    return response.json() as Promise<T>;
  } catch (error: unknown) {
    const err = error as Error;
    const isNetworkFailure =
      err instanceof TypeError ||
      err.message === "Failed to fetch" ||
      err.name === "AbortError";
    if (isNetworkFailure) {
      const hint =
        `Cannot reach the backend at ${API_BASE_URL}. ` +
        "Start it with: backend\\venv\\Scripts\\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000";
      console.error(`API Fetch Error [${endpoint}]:`, hint, err);
      throw new Error(hint);
    }
    console.error(`API Fetch Error [${endpoint}]:`, err);
    throw err;
  }
}

export const api = {
  getHealth: () => apiFetch<{ status: string; database: string; timestamp: string }>("/api/health"),
  
  createCase: (notes?: string, userId?: string) => 
    apiFetch<Case>("/api/cases", {
      method: "POST",
      body: JSON.stringify({ user_id: userId, notes }),
    }),
    
  getCase: (caseId: string) => 
    apiFetch<Case>(`/api/cases/${caseId}`),
    
  listCases: (limit = 20) => 
    apiFetch<Case[]>(`/api/cases?limit=${limit}`),
    
  submitQuestionnaire: (caseId: string, payload: any) =>
    apiFetch<any>(`/api/cases/${caseId}/questionnaire`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  getQuestionnaireTemplate: (caseId: string) =>
    apiFetch<any>(`/api/cases/${caseId}/questionnaire/template`),
    
  getModels: () => 
    apiFetch<any[]>("/api/models"),

  getModelRegistry: () =>
    apiFetch<Record<string, any>>("/api/models/registry"),
    
  getEvaluation: () => 
    apiFetch<any>("/api/evaluation"),

  getComparison: () => 
    apiFetch<any>("/api/evaluation/comparison"),

  getAblation: () => 
    apiFetch<any>("/api/evaluation/ablation"),

  abortSOS: (caseId: string) => 
    apiFetch<any>(`/api/cases/${caseId}/sos/abort`, { method: "POST" }),

  getSOSStatus: (caseId: string) => 
    apiFetch<any>(`/api/cases/${caseId}/sos/status`),

  getSOSConfig: () =>
    apiFetch<any>("/api/sos/config"),


  triggerDemoSOS: (
    caseId: string,
    mode: "local_demo" | "twilio_test" = "local_demo",
    geo?: {
      latitude?: number;
      longitude?: number;
      accuracy_m?: number;
      location_label?: string;
      maps_url?: string;
    } | null
  ) =>
    apiFetch<any>(`/api/cases/${caseId}/sos/demo/trigger`, {
      method: "POST",
      body: JSON.stringify({ mode, ...(geo || {}) }),
    }),

  respondDemoSOS: (
    caseId: string,
    userResponse: "safe" | "no_response",
    mode: "local_demo" | "twilio_test" = "local_demo",
    geo?: {
      latitude?: number;
      longitude?: number;
      accuracy_m?: number;
      location_label?: string;
      maps_url?: string;
    } | null
  ) =>
    apiFetch<any>(`/api/cases/${caseId}/sos/demo/respond`, {
      method: "POST",
      body: JSON.stringify({ user_response: userResponse, mode, ...(geo || {}) }),
    }),


  loadDemoSensor: (caseId: string) =>
    apiFetch<any>(`/api/cases/${caseId}/sensor/demo`, { method: "POST" }),

  skipSensor: (caseId: string) =>
    apiFetch<any>(`/api/cases/${caseId}/sensor/skip`, { method: "POST" }),

  simulateSensor: (caseId: string, scenario: string) =>
    apiFetch<any>(`/api/cases/${caseId}/sensor/simulate`, {
      method: "POST",
      body: JSON.stringify({ scenario }),
    }),

  uploadLiveSensor: (caseId: string, payload: any) =>
    apiFetch<any>(`/api/cases/${caseId}/sensor/live/upload`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  runCompleteDemo: () =>
    apiFetch<Case>("/api/cases/demo", { method: "POST" }),

  getReport: (caseId: string) =>
    apiFetch<any>(`/api/cases/${caseId}/report`),
};
