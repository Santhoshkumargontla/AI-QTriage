"use client";

import React, { useState } from "react";
import { Camera, AlertTriangle, Eye, Layers, Maximize2, SplitSquareVertical } from "lucide-react";
import { getUploadUrl } from "@/lib/api";

interface ImageAnalysisProps {
  caseData: any;
  liveYoloClasses?: string[];
}

export function ImageAnalysis({ caseData, liveYoloClasses = [] }: ImageAnalysisProps) {
  const [activeViewTab, setActiveViewTab] = useState<"detection" | "gradcam" | "segmentation" | "side_by_side">("detection");
  const vis = caseData?.visible_injury;

  // YOLO11 Finding & Confidence
  const yoloFinding = vis?.finding || vis?.yolo_finding || (vis?.yolo_finding_detected === false ? "None detected" : "Unavailable");
  const rawYoloConf = vis?.confidence != null ? Number(vis.confidence) : (vis?.yolo_confidence != null ? Number(vis.yolo_confidence) : null);
  const yoloConfStr = rawYoloConf != null ? `${(rawYoloConf * 100).toFixed(1)}%` : "Unavailable";

  // EfficientNetV2 Finding & Probability
  const isOod = vis?.classifier_status === "OUT_OF_DISTRIBUTION";
  const clfFinding = isOod ? "Withheld — OOD detected" : (vis?.classifier_finding || vis?.classifier_status || "Unavailable");
  const rawClfProb = vis?.classifier_probability != null ? Number(vis.classifier_probability) : null;
  const clfProbDisplay = isOod 
    ? "Probability: Not available" 
    : (rawClfProb != null ? `${(rawClfProb * 100).toFixed(1)}% probability` : "Probability: Unavailable");

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
    disagreementMessage = `Vision models disagree: YOLO11 (${yoloFinding} · ${yoloConfStr}) vs. EfficientNetV2 (${clfFinding} · ${clfProbDisplay}). Automated interpretation requires human review.`;
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

  // Helper to render YOLO Canvas
  const renderYoloCanvas = (maxW = "540px") => (
    <div
      className="relative bg-slate-950 rounded-xl overflow-hidden border border-slate-800 mx-auto w-full shadow-inner"
      style={{
        maxWidth: maxW,
        aspectRatio: `${refW} / ${refH}`,
      }}
    >
      {originalImageUrl ? (
        <>
          <img
            src={originalImageUrl}
            alt="Original camera upload with detections"
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
          Original Image Artifact Unavailable
        </div>
      )}
    </div>
  );

  // Helper to render Grad-CAM Canvas
  const renderGradcamCanvas = (maxW = "540px") => (
    <div
      className="relative bg-slate-950 rounded-xl overflow-hidden border border-slate-800 mx-auto w-full flex items-center justify-center shadow-inner"
      style={{
        maxWidth: maxW,
        aspectRatio: `${refW} / ${refH}`,
      }}
    >
      {gradcamOverlayUrl ? (
        <>
          <img
            src={gradcamOverlayUrl}
            alt="Grad-CAM Saliency Heatmap"
            className="absolute inset-0 w-full h-full object-contain"
          />
          <div className="absolute top-2 right-2 bg-slate-900/80 backdrop-blur-md px-2 py-0.5 rounded text-[10px] font-mono text-amber-300 border border-amber-500/30">
            MODEL VISUALIZATION — NOT CLINICAL
          </div>
        </>
      ) : (
        <div className="p-6 text-center space-y-2 max-w-sm">
          <div className="mx-auto w-10 h-10 rounded-full bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400">
            <Layers className="h-5 w-5" />
          </div>
          <p className="text-xs font-bold text-amber-300">
            Grad-CAM Heatmap Withheld
          </p>
          <p className="text-[11px] text-slate-400 leading-relaxed">
            {isOod
              ? "EfficientNetV2 whole-image classifier flagged input as OUT_OF_DISTRIBUTION. Saliency maps are withheld to avoid false clinical reassurance."
              : "Grad-CAM overlay not generated for this case record."}
          </p>
          <span className="inline-block text-[10px] font-mono px-2.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400">
            Status: {vis?.gradcam_explanation_status || "WITHHELD"}
          </span>
        </div>
      )}
    </div>
  );

  // Helper to render Segmentation Canvas
  const renderMaskCanvas = (maxW = "540px") => (
    <div
      className="relative bg-slate-950 rounded-xl overflow-hidden border border-slate-800 mx-auto w-full flex items-center justify-center shadow-inner"
      style={{
        maxWidth: maxW,
        aspectRatio: `${refW} / ${refH}`,
      }}
    >
      {maskUrl ? (
        <>
          <img
            src={maskUrl}
            alt="U-Net Binary Segmentation Mask"
            className="absolute inset-0 w-full h-full object-contain"
          />
          <div className="absolute top-2 right-2 bg-slate-900/80 backdrop-blur-md px-2 py-0.5 rounded text-[10px] font-mono text-purple-300 border border-purple-500/30">
            U-NET BINARY MASK
          </div>
        </>
      ) : (
        <div className="p-6 text-center space-y-2 max-w-sm">
          <div className="mx-auto w-10 h-10 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-400">
            <Layers className="h-5 w-5" />
          </div>
          <p className="text-xs font-bold text-slate-300">
            Segmentation Mask Unavailable
          </p>
          <p className="text-[11px] text-slate-500">
            Binary lesion surface mask was not extracted or did not pass quality sanity thresholds.
          </p>
        </div>
      )}
    </div>
  );

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

      {/* Three Clickable Interactive Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {/* YOLO11 Card */}
        <button
          type="button"
          onClick={() => setActiveViewTab("detection")}
          className={`p-4 rounded-xl border text-center space-y-1 shadow-md transition-all text-left w-full flex flex-col justify-between ${
            activeViewTab === "detection"
              ? "bg-slate-900 border-blue-500 ring-2 ring-blue-500/20"
              : "bg-slate-900/80 border-slate-800 hover:border-slate-700 hover:bg-slate-850"
          }`}
        >
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider block">
                YOLO11 Spatial Detection
              </span>
              <span className="text-[9px] font-mono font-bold text-blue-400 bg-blue-500/10 px-1.5 py-0.5 rounded">
                Click to view
              </span>
            </div>
            <span className="text-lg font-black text-white block capitalize">
              {yoloFinding}
            </span>
            <span className="text-xs font-mono font-bold text-emerald-400 block">
              {yoloConfStr} confidence
            </span>
          </div>
          <p className="text-[10px] text-slate-400 leading-tight pt-1.5 border-t border-slate-800/80 mt-1">
            Note: Summary displays the leading spatial detection; the image overlay may contain multiple detected regions.
          </p>
        </button>

        {/* EfficientNetV2 Card */}
        <button
          type="button"
          onClick={() => setActiveViewTab("gradcam")}
          className={`p-4 rounded-xl border text-center space-y-1 shadow-md transition-all text-left w-full flex flex-col justify-between ${
            activeViewTab === "gradcam"
              ? "bg-slate-900 border-cyan-500 ring-2 ring-cyan-500/20"
              : "bg-slate-900/80 border-slate-800 hover:border-slate-700 hover:bg-slate-850"
          }`}
        >
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider block">
                EfficientNetV2 Classifier
              </span>
              <span className="text-[9px] font-mono font-bold text-cyan-400 bg-cyan-500/10 px-1.5 py-0.5 rounded">
                Click to view
              </span>
            </div>
            <span className={`text-lg font-black block capitalize ${isOod ? "text-amber-400" : "text-white"}`}>
              {clfFinding}
            </span>
            <span className="text-xs font-mono font-bold text-cyan-400 block">
              {clfProbDisplay}
            </span>
          </div>
          <p className="text-[10px] text-slate-400 leading-tight pt-1.5 border-t border-slate-800/80 mt-1">
            {isOod ? "Input exceeds domain confidence threshold. Classifier predictions and saliency maps withheld." : "Whole-image classification hypothesis."}
          </p>
        </button>

        {/* ResNet34-U-Net Card */}
        <button
          type="button"
          onClick={() => setActiveViewTab("segmentation")}
          className={`p-4 rounded-xl border text-center space-y-1 shadow-md transition-all text-left w-full flex flex-col justify-between ${
            activeViewTab === "segmentation"
              ? "bg-slate-900 border-purple-500 ring-2 ring-purple-500/20"
              : "bg-slate-900/80 border-slate-800 hover:border-slate-700 hover:bg-slate-850"
          }`}
        >
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider block">
                ResNet34-U-Net Segmentation
              </span>
              <span className="text-[9px] font-mono font-bold text-purple-400 bg-purple-500/10 px-1.5 py-0.5 rounded">
                Click to view
              </span>
            </div>
            <span className={`text-lg font-black block ${segValid ? "text-white" : "text-slate-400"}`}>
              {segStatusStr}
            </span>
            <span className="text-xs font-mono font-bold text-purple-400 block">
              Affected ROI: {segAreaStr}
            </span>
          </div>
          <p className="text-[10px] text-slate-400 leading-tight pt-1.5 border-t border-slate-800/80 mt-1">
            Binary lesion pixel mask segmented across region of interest.
          </p>
        </button>
      </div>

      {/* Visual Artifacts Display: Dynamic Output Controller */}
      <div className="p-5 rounded-2xl bg-slate-900/70 border border-slate-800 space-y-4">
        {/* Interactive Tab Switcher */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <div className="space-y-0.5">
            <span className="text-xs font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
              <Eye className="h-4 w-4 text-emerald-400" />
              <span>Visual Inspection Workspace</span>
            </span>
            <p className="text-[11px] text-slate-400">
              Click any output mode below to render its dedicated detection or attention layer.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-1.5 bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs self-start sm:self-auto">
            <button
              type="button"
              onClick={() => setActiveViewTab("detection")}
              className={`px-3 py-1.5 rounded-lg font-bold transition-all flex items-center gap-1.5 ${
                activeViewTab === "detection"
                  ? "bg-blue-600 text-white shadow-md shadow-blue-600/30"
                  : "text-slate-400 hover:text-white hover:bg-slate-900"
              }`}
            >
              <Camera className="h-3.5 w-3.5" />
              <span>YOLO11 Overlay</span>
            </button>
            <button
              type="button"
              onClick={() => setActiveViewTab("gradcam")}
              className={`px-3 py-1.5 rounded-lg font-bold transition-all flex items-center gap-1.5 ${
                activeViewTab === "gradcam"
                  ? "bg-cyan-600 text-white shadow-md shadow-cyan-600/30"
                  : "text-slate-400 hover:text-white hover:bg-slate-900"
              }`}
            >
              <Layers className="h-3.5 w-3.5" />
              <span>Grad-CAM Heatmap</span>
            </button>
            <button
              type="button"
              onClick={() => setActiveViewTab("segmentation")}
              className={`px-3 py-1.5 rounded-lg font-bold transition-all flex items-center gap-1.5 ${
                activeViewTab === "segmentation"
                  ? "bg-purple-600 text-white shadow-md shadow-purple-600/30"
                  : "text-slate-400 hover:text-white hover:bg-slate-900"
              }`}
            >
              <Maximize2 className="h-3.5 w-3.5" />
              <span>U-Net Mask</span>
            </button>
            <button
              type="button"
              onClick={() => setActiveViewTab("side_by_side")}
              className={`px-3 py-1.5 rounded-lg font-bold transition-all flex items-center gap-1.5 ${
                activeViewTab === "side_by_side"
                  ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
                  : "text-slate-400 hover:text-white hover:bg-slate-900"
              }`}
            >
              <SplitSquareVertical className="h-3.5 w-3.5" />
              <span>Side-by-Side</span>
            </button>
          </div>
        </div>

        {/* OUTPUT 1: YOLO11 Dedicated Overlay View */}
        {activeViewTab === "detection" && (
          <div className="space-y-4 animate-fadeIn">
            <div className="flex items-center justify-between text-xs text-slate-300">
              <span className="font-bold flex items-center gap-2">
                <span className="inline-block w-2.5 h-2.5 rounded-full bg-blue-500"></span>
                <span>Active Output: YOLO11 Object Localization Bounding Boxes</span>
              </span>
              <span className="font-mono text-[11px] text-slate-400">
                Resolution: {refW} &times; {refH} px
              </span>
            </div>

            {renderYoloCanvas("580px")}

            {/* Detections Output Table / List */}
            <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800 space-y-2">
              <span className="text-[11px] font-bold text-slate-300 uppercase tracking-wider block">
                YOLO11 Detected Lesion Boxes ({detectionsToDraw.length} items)
              </span>
              {detectionsToDraw.length > 0 ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2 text-xs">
                  {detectionsToDraw.map((det: any, i: number) => (
                    <div key={i} className="p-2.5 bg-slate-900 rounded-lg border border-slate-800 space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-white capitalize">{det.finding}</span>
                        <span className="font-mono text-emerald-400 font-bold">{((det.confidence || 0) * 100).toFixed(1)}%</span>
                      </div>
                      {det.bounding_box && (
                        <p className="font-mono text-[10px] text-slate-400 truncate">
                          Coords: [{det.bounding_box.map((n: number) => Math.round(n)).join(", ")}]
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-slate-500 italic">No spatial bounding boxes detected above threshold.</p>
              )}
            </div>
          </div>
        )}

        {/* OUTPUT 2: Grad-CAM Dedicated Heatmap View */}
        {activeViewTab === "gradcam" && (
          <div className="space-y-4 animate-fadeIn">
            <div className="flex items-center justify-between text-xs text-slate-300">
              <span className="font-bold flex items-center gap-2">
                <span className="inline-block w-2.5 h-2.5 rounded-full bg-cyan-500"></span>
                <span>Active Output: EfficientNetV2 Grad-CAM Saliency Attention Map</span>
              </span>
              <span className="font-mono text-[11px] text-slate-400">
                Layer: Target Conv Feature Map
              </span>
            </div>

            {renderGradcamCanvas("580px")}

            <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800 space-y-2 text-xs">
              <span className="text-[11px] font-bold text-slate-300 uppercase tracking-wider block">
                Grad-CAM Interpretation &amp; Model Visualization Details
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-slate-300">
                <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800 space-y-1">
                  <span className="text-[10px] font-bold text-slate-400 uppercase block">Hypothesis Label</span>
                  <span className="text-sm font-bold text-cyan-300 block capitalize">{clfFinding}</span>
                  <span className="text-[10px] text-slate-400 font-mono">Classifier status: {vis?.classifier_status || "UNKNOWN"}</span>
                </div>
                <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800 space-y-1">
                  <span className="text-[10px] font-bold text-slate-400 uppercase block">Explanation Integrity</span>
                  <span className="text-sm font-bold text-white block">
                    {gradcamOverlayUrl ? "Heatmap Overlay Active" : "Heatmap Withheld (OOD Guard)"}
                  </span>
                  <span className="text-[10px] text-slate-400 font-mono">Research visualization only — not clinical ground truth</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* OUTPUT 3: ResNet34-U-Net Segmentation Mask View */}
        {activeViewTab === "segmentation" && (
          <div className="space-y-4 animate-fadeIn">
            <div className="flex items-center justify-between text-xs text-slate-300">
              <span className="font-bold flex items-center gap-2">
                <span className="inline-block w-2.5 h-2.5 rounded-full bg-purple-500"></span>
                <span>Active Output: ResNet34-U-Net Binary Segmentation Mask</span>
              </span>
              <span className="font-mono text-[11px] text-slate-400">
                Status: {segStatusStr}
              </span>
            </div>

            {renderMaskCanvas("580px")}

            <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800 space-y-2 text-xs">
              <span className="text-[11px] font-bold text-slate-300 uppercase tracking-wider block">
                Segmentation Surface Area Metrics
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-slate-300">
                <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800 text-center">
                  <span className="text-[10px] font-bold text-slate-400 uppercase block">Affected ROI</span>
                  <span className="text-lg font-black text-purple-400 font-mono">{segAreaStr}</span>
                </div>
                <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800 text-center">
                  <span className="text-[10px] font-bold text-slate-400 uppercase block">Positive Pixels</span>
                  <span className="text-lg font-black text-white font-mono">
                    {vis?.segmentation_debug?.positive_pixels != null ? `${vis.segmentation_debug.positive_pixels.toLocaleString()} px` : "Unavailable"}
                  </span>
                </div>
                <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800 text-center">
                  <span className="text-[10px] font-bold text-slate-400 uppercase block">Full Frame Ratio</span>
                  <span className="text-lg font-black text-cyan-400 font-mono">
                    {vis?.segmentation_debug?.positive_pixels != null ? `${((vis.segmentation_debug.positive_pixels / (refW * refH)) * 100).toFixed(2)}%` : "Unavailable"}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* OUTPUT 4: Side-by-Side Dual Comparison View */}
        {activeViewTab === "side_by_side" && (
          <div className="space-y-4 animate-fadeIn">
            <div className="flex items-center justify-between text-xs text-slate-300">
              <span className="font-bold flex items-center gap-2">
                <span className="inline-block w-2.5 h-2.5 rounded-full bg-indigo-500"></span>
                <span>Active Output: Dual Side-by-Side Comparison (YOLO11 Detection vs. Grad-CAM Attention)</span>
              </span>
              <span className="font-mono text-[11px] text-slate-400">
                Synchronized Geometry
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <span className="text-xs font-bold text-slate-300 block">YOLO11 Object Localization</span>
                {renderYoloCanvas("480px")}
              </div>
              <div className="space-y-2">
                <span className="text-xs font-bold text-slate-300 block">Grad-CAM Saliency Heatmap</span>
                {renderGradcamCanvas("480px")}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
