"use client";

import React from "react";
import { AlertTriangle, Clock, Database, ShieldAlert } from "lucide-react";

interface ReviewStatusBannerProps {
  caseData: any;
  caseId: string;
}

export function ReviewStatusBanner({ caseData, caseId }: ReviewStatusBannerProps) {
  const isDemo = Boolean(
    caseData?.is_demo ||
    caseData?.visible_injury?.source_type === "demo" ||
    caseData?.sensor_summary?.source_type === "demo" ||
    caseData?.sensor_data_is_demo
  );

  const formattedDate = caseData?.created_at
    ? new Date(caseData.created_at).toLocaleString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      })
    : "Not provided";

  return (
    <div className="space-y-4">
      {/* Top Header Card */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-5 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl">
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-mono font-bold tracking-wider text-slate-400 uppercase">
              Case Assessment
            </span>
            {isDemo && (
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-purple-500/10 text-purple-300 border border-purple-500/20 flex items-center gap-1 font-mono">
                <Database className="h-3 w-3" />
                SYNTHETIC DEMONSTRATION
              </span>
            )}
            <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-amber-500/10 text-amber-300 border border-amber-500/20 flex items-center gap-1">
              <AlertTriangle className="h-3 w-3 text-amber-400" />
              HUMAN REVIEW REQUIRED
            </span>
          </div>
          <h1 className="text-xl sm:text-2xl font-black text-white tracking-tight font-mono">
            {caseId}
          </h1>
        </div>

        <div className="flex flex-wrap items-center gap-3 text-xs text-slate-400">
          <div className="flex items-center gap-1.5 bg-slate-950/60 px-3 py-1.5 rounded-xl border border-slate-800 font-mono text-[11px]">
            <Clock className="h-3.5 w-3.5 text-slate-500" />
            <span>Completed: {formattedDate}</span>
          </div>
        </div>
      </div>

      {/* Prominent Consolidated Safety Review Banner */}
      <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-600/40 text-amber-200 text-xs sm:text-sm font-medium flex items-start sm:items-center gap-3 shadow-lg">
        <ShieldAlert className="h-5 w-5 text-amber-400 shrink-0 mt-0.5 sm:mt-0" />
        <div className="space-y-0.5">
          <p className="font-bold text-amber-300 tracking-wide uppercase text-xs">
            Clinical Governance Gate Active
          </p>
          <p className="text-amber-200/90 leading-relaxed text-xs">
            Human review required. Automated assessment is not a clinical diagnosis. This research prototype has not been clinically validated.
          </p>
        </div>
      </div>
    </div>
  );
}
