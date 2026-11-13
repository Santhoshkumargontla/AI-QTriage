"use client";

import React, { useState } from "react";
import { Camera, AlertTriangle, Eye, Layers } from "lucide-react";
import { getUploadUrl } from "@/lib/api";

interface ImageAnalysisProps {
  caseData: any;
  liveYoloClasses?: string[];
}

export function ImageAnalysis({ caseData, liveYoloClasses = [] }: ImageAnalysisProps) {
  const [activeViewTab, setActiveViewTab] = useState<"detection" | "gradcam" | "segmentation">("detection");
  const vis = caseData?.visible_injury;

  // YOLO11 Finding & Confidence
  const yoloFinding = vis?.finding || vis?.yolo_finding || (vis?.yolo_finding_detected === false ? "None detected" : "Unavailable");
  const rawYoloConf = vis?.confidence != null ? Number(vis.confidence) : (vis?.yolo_confidence != null ? Number(vis.yolo_confidence) : null);
  const yoloConfStr = rawYoloConf != null ? `${(rawYoloConf * 100).toFixed(1)}%` : "Unavailable";

  // EfficientNetV2 Finding & Probability
  const isOod = vis?.classifier_status === "OUT_OF_DISTRIBUTION";
  const clfFinding = vis?.classifier_finding || (isOod ? "Withheld – OOD" : (vis?.classifier_status || "Unavailable"));
  const rawClfProb = vis?.classifier_probability != null ? Number(vis.classifier_probability) : null;
  const clfProbStr = rawClfProb != null ? `${(rawClfProb * 100).toFixed(1)}%` : (isOod ? "Withheld (OOD)" : "Unavailable");

  // ResNet34-U-Net Segmentation Status & Area
  const segValid = vis?.segmentation_status === "VALID" && vis?.affected_ratio != null;
  const segAreaStr = segValid ? `${(Number(vis.affected_ratio) * 100).toFixed(1)}% (ROI)` : "Unavailable";
  const segStatusStr = segValid ? "Mask Valid" : (vis?.segmentation_status || "Unavailable");

  // Vision Disagreement Assessment
  const disagreementDetected = Boolean(
    vis?.vision_disagreement ||
    vis?.disagreement_policy?.disagreement_detected ||
    (vis?.finding && vis?.classifier_finding && String(vis.finding).toLowerCase() !== String(vis.classifier_finding).toLowerCase()) ||
    isOod
  );

  let disagreementMessage = "";
  if (isOod) {
    disagreementMessage = `Vision models disagree: YOLO11 detected ${yoloFinding} (${yoloConfStr}) while EfficientNetV2 flagged OUT_OF_DISTRIBUTION. Automated interpretation requires human review.`;
  } else if (vis?.finding && vis?.classifier_finding) {
    disagreementMessage = `Vision models disagree: YOLO11 (${yoloFinding} · ${yoloConfStr}) vs. EfficientNetV2 (${clfFinding} · ${clfProbStr}). Automated interpretation requires human review.`;
  } else if (disagreementDetected) {
    disagreementMessage = "Vision models disagree: Multiple conflicting hypotheses detected across visual pipelines. Automated interpretation requires human review.";
  }

  // Visual Assets
  const originalImageUrl = vis?.image_url ? getUploadUrl(vis.image_url) : null;
  const gradcamOverlayUrl = vis?.overlay_url && vis?.gradcam_overlay_generated ? getUploadUrl(vis.overlay_url) : null;
  const maskUrl = vis?.mask_url ? getUploadUrl(vis.mask_url) : null;

  // Bounding box coordinate calculations
  const refW = vis?.original_width || vis?.overlay_width || 1283;
  const refH = vis?.original_height || vis?.overlay_height || 700;
  const detectionsToDraw = Array.isArray(vis?.all_detections) && vis.all_detections.length > 0
    ? vis.all_detections
    : (vis?.bounding_box && vis?.yolo_finding_detected
        ? [{ finding: yoloFinding, confidence: rawYoloConf, bounding_box: vis.bounding_box }]
        : []);

  const colorMap: Record<string, { border: string; bg: string; text: string }> = {
    cut: { border: "border-red-500", bg: "bg-red-500/15", text: "bg-red-600" },
    laceration: { border: "border-red-500", bg: "bg-red-500/15", text: "bg-red-600" },
    wound: { border: "border-rose-500", bg: "bg-rose-500/15", text: "bg-rose-600" },
    abrasion: { border: "border-orange-500", bg: "bg-orange-500/15", text: "bg-orange-600" },
    bruise: { border: "border-amber-500", bg: "bg-amber-500/15", text: "bg-amber-600" },
    burn: { border: "border-purple-500", bg: "bg-purple-500/15", text: "bg-purple-600" },
  };

  return (
    <div className="space-y-4">
      {/* Section Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center gap-2">
          <Camera className="h-5 w-5 text-emerald-400" />
          <h2 className="text-base font-bold text-white uppercase tracking-wider">
            Section 2: Image Analysis
          </h2>
        </div>
        <span className="text-xs text-slate-400 font-mono">
          YOLO11 + EfficientNetV2 + U-Net
        </span>
      </div>

      {/* Disagreement Alert Banner */}
      {disagreementDetected && (
        <div className="p-3.5 rounded-xl bg-rose-950/40 border border-rose-600/40 text-rose-200 text-xs sm:text-sm font-medium flex items-start gap-2.5 shadow-md">
          <AlertTriangle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
          <div className="space-y-0.5">
            <span className="font-bold text-rose-300 block uppercase text-[11px] tracking-wide">
              Vision Hypothesis Conflict
            </span>
            <p className="leading-relaxed text-xs">{disagreementMessage}</p>
          </div>
        </div>
      )}

      {/* Three Compact Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {/* YOLO11 Card */}
        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 text-center space-y-1 shadow-md">
          <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider block">
            YOLO11 Spatial Detection
          </span>
          <span className="text-lg font-black text-white block capitalize">
            {yoloFinding}
          </span>
          <span className="text-xs font-mono font-bold text-emerald-400 block">
            {yoloConfStr} confidence
          </span>
        </div>

        {/* EfficientNetV2 Card */}
        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 text-center space-y-1 shadow-md">
          <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider block">
            EfficientNetV2 Classifier
          </span>
          <span className={`text-lg font-black block capitalize ${isOod ? "text-amber-400" : "text-white"}`}>
            {clfFinding}
          </span>
          <span className="text-xs font-mono font-bold text-cyan-400 block">
            {clfProbStr} probability
          </span>
        </div>

        {/* ResNet34-U-Net Card */}
        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 text-center space-y-1 shadow-md">
          <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider block">
            ResNet34-U-Net Segmentation
          </span>
          <span className={`text-lg font-black block ${segValid ? "text-white" : "text-slate-400"}`}>
            {segStatusStr}
          </span>
          <span className="text-xs font-mono font-bold text-purple-400 block">
            Affected Area: {segAreaStr}
          </span>
        </div>
      </div>

      {/* Visual Artifacts Display: Side-by-Side Dual Visualizer */}
      <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-2">
          <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
            <Eye className="h-4 w-4 text-emerald-400" />
            Visual Evidence &amp; Model Explanations
          </span>
          <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-[11px] self-start sm:self-auto">
            <button
              type="button"
              onClick={() => setActiveViewTab("detection")}
              className={`px-2.5 py-1 rounded font-medium transition-all ${
                activeViewTab === "detection"
                  ? "bg-blue-600 text-white font-bold"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              YOLO11 Overlay
            </button>
            <button
              type="button"
              onClick={() => setActiveViewTab("gradcam")}
              className={`px-2.5 py-1 rounded font-medium transition-all ${
                activeViewTab === "gradcam"
                  ? "bg-blue-600 text-white font-bold"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Grad-CAM Heatmap
            </button>
            {maskUrl && (
              <button
                type="button"
                onClick={() => setActiveViewTab("segmentation")}
                className={`px-2.5 py-1 rounded font-medium transition-all ${
                  activeViewTab === "segmentation"
                    ? "bg-blue-600 text-white font-bold"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                U-Net Mask
              </button>
            )}
          </div>
        </div>

        {/* Dual Responsive Canvas Layout */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
          {/* Panel 1: Original Image with YOLO11 Detection Overlay */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-[11px]">
              <span className="font-bold text-slate-300">YOLO11 Object Localization</span>
              <span className="text-slate-400 font-mono">Original XYXY space</span>
            </div>
            <div
              className="relative bg-slate-950 rounded-xl overflow-hidden border border-slate-800 mx-auto w-full"
              style={{
                maxWidth: "480px",
                aspectRatio: `${refW} / ${refH}`,
              }}
            >
              {originalImageUrl ? (
                <>
                  <img
                    src={originalImageUrl}
                    alt="Original camera upload"
                    className="absolute inset-0 w-full h-full object-contain"
                  />
                  {detectionsToDraw.map((det: any, idx: number) => {
                    const box = det.bounding_box;
                    if (!box || box.length < 4) return null;
                    const findingKey = String(det.finding || "").toLowerCase();
                    const colors = colorMap[findingKey] || { border: "border-cyan-500", bg: "bg-cyan-500/15", text: "bg-cyan-600" };
                    const label = String(det.finding || "Injury").toUpperCase();
                    const confPct = ((det.confidence || 0) * 100).toFixed(0);

                    return (
                      <div
                        key={idx}
                        className={`absolute border-2 ${colors.border} ${colors.bg} rounded pointer-events-none transition-all`}
                        style={{
                          left: `${(box[0] / refW) * 100}%`,
                          top: `${(box[1] / refH) * 100}%`,
                          width: `${Math.min(100, Math.max(0, ((box[2] - box[0]) / refW) * 100))}%`,
                          height: `${Math.min(100, Math.max(0, ((box[3] - box[1]) / refH) * 100))}%`,
                        }}
                      >
                        <span className={`absolute top-0 left-0 ${colors.text} text-white text-[9px] px-1 rounded-br font-bold shadow-md`}>
                          {label} ({confPct}%)
                        </span>
                      </div>
                    );
                  })}
                </>
              ) : (
                <div className="flex items-center justify-center h-full text-slate-500 text-xs font-mono">
                  Image Artifact Unavailable
                </div>
              )}
            </div>
          </div>

          {/* Panel 2: Grad-CAM Visualization or Active Artifact */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-[11px]">
              <span className="font-bold text-slate-300">
                {activeViewTab === "gradcam"
                  ? "Grad-CAM Saliency Map"
                  : activeViewTab === "segmentation"
                  ? "ResNet34-U-Net Segmentation"
                  : "EfficientNetV2 Attention Layer"}
              </span>
              <span className="text-slate-400 font-mono text-[10px]">
                {activeViewTab === "gradcam" && gradcamOverlayUrl ? "MODEL VISUALIZATION" : "ARTIFACT STATUS"}
              </span>
            </div>
            <div
              className="relative bg-slate-950 rounded-xl overflow-hidden border border-slate-800 mx-auto w-full flex items-center justify-center"
              style={{
                maxWidth: "480px",
                aspectRatio: `${refW} / ${refH}`,
              }}
            >
              {activeViewTab === "segmentation" && maskUrl ? (
                <img
                  src={maskUrl}
                  alt="U-Net Binary Segmentation Mask"
                  className="absolute inset-0 w-full h-full object-contain"
                />
              ) : gradcamOverlayUrl ? (
                <>
                  <img
                    src={gradcamOverlayUrl}
                    alt="Grad-CAM Saliency Heatmap"
                    className="absolute inset-0 w-full h-full object-contain"
                  />
                  <div className="absolute top-2 right-2 bg-slate-900/80 backdrop-blur-md px-2 py-0.5 rounded text-[10px] font-mono text-amber-300 border border-amber-500/30">
                    MODEL VISUALIZATION
                  </div>
                </>
              ) : (
                <div className="p-6 text-center space-y-2 max-w-sm">
                  <div className="mx-auto w-8 h-8 rounded-full bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400">
                    <Layers className="h-4 w-4" />
                  </div>
                  <p className="text-xs font-bold text-amber-300">
                    Grad-CAM Heatmap Withheld
                  </p>
                  <p className="text-[11px] text-slate-400 leading-relaxed">
                    {isOod
                      ? "EfficientNetV2 flagged input as OUT_OF_DISTRIBUTION. Misleading saliency maps are withheld to avoid false clinical reassurance."
                      : "Saliency overlay not generated for this case record."}
                  </p>
                  <span className="inline-block text-[10px] font-mono px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400">
                    Status: {vis?.gradcam_explanation_status || "WITHHELD"}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
