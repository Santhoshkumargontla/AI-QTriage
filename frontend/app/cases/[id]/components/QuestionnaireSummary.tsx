"use client";

import React from "react";
import { FileText } from "lucide-react";

interface QuestionnaireSummaryProps {
  caseData: any;
}

export function QuestionnaireSummary({ caseData }: QuestionnaireSummaryProps) {
  const answers = caseData?.questionnaire?.answers || {};

  // Dynamic field mappings without hardcoded substitutions
  const painLevel = answers.pain_level != null ? `${answers.pain_level} / 10` : "Not provided";
  const injuryMechanism = answers.injury_mechanism || answers.cause || "Not provided";
  const movementLimitation = answers.movement_limitation != null ? String(answers.movement_limitation) : "Not provided";
  
  let weightBearing = "Not provided";
  if (answers.weight_bearing != null) {
    if (typeof answers.weight_bearing === "boolean") {
      weightBearing = answers.weight_bearing ? "Yes" : "Unable (No)";
    } else {
      weightBearing = String(answers.weight_bearing);
    }
  }

  const visibleBleeding = answers.visible_bleeding || answers.bleeding || "Not provided";
  
  let crackPop = "Not provided";
  if (answers.crack_pop != null) {
    if (typeof answers.crack_pop === "boolean") {
      crackPop = answers.crack_pop ? "Yes" : "No";
    } else {
      crackPop = String(answers.crack_pop);
    }
  }

  let directImpact = "Not provided";
  if (answers.direct_impact != null) {
    if (typeof answers.direct_impact === "boolean") {
      directImpact = answers.direct_impact ? "Yes" : "No";
    } else {
      directImpact = String(answers.direct_impact);
    }
  } else if (answers.cause === "direct_blow") {
    directImpact = "Yes";
  }

  const items = [
    { label: "Pain Intensity", value: painLevel, highlight: answers.pain_level != null && Number(answers.pain_level) >= 7 },
    { label: "Injury Mechanism", value: injuryMechanism, capitalize: true },
    { label: "Movement Limitation", value: movementLimitation, capitalize: true },
    { label: "Weight-Bearing Ability", value: weightBearing, capitalize: true },
    { label: "Visible Bleeding", value: visibleBleeding, capitalize: true },
    { label: "Crack / Pop Sensation", value: crackPop, highlight: crackPop.toLowerCase() === "yes" },
    { label: "Direct Impact Vector", value: directImpact, capitalize: true },
  ];

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center gap-2">
          <FileText className="h-5 w-5 text-emerald-400" />
          <h2 className="text-base font-bold text-white uppercase tracking-wider">
            Questionnaire Context
          </h2>
        </div>
        <span className="text-xs text-slate-400 font-mono">
          Structured Clinical Survey
        </span>
      </div>

      {/* Compact Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2.5">
        {items.map((item, idx) => (
          <div
            key={idx}
            className={`p-3 rounded-xl border bg-slate-900/80 space-y-1 ${
              item.highlight
                ? "border-amber-600/50 bg-amber-950/20 text-amber-200"
                : "border-slate-800 text-slate-300"
            }`}
          >
            <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider truncate">
              {item.label}
            </span>
            <span
              className={`text-sm font-bold block truncate ${
                item.highlight
                  ? "text-amber-300 font-black"
                  : item.value === "Not provided"
                  ? "text-slate-500 italic font-normal"
                  : "text-white"
              } ${item.capitalize ? "capitalize" : ""}`}
            >
              {item.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
