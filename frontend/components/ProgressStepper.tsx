"use client";

import { Check } from "lucide-react";

interface ProgressStepperProps {
  currentStep: number;
}

const STEPS = [
  { id: 1, label: "Upload Image", short: "Upload" },
  { id: 2, label: "Image Analysis", short: "Vision" },
  { id: 3, label: "Questionnaire", short: "Form" },
  { id: 4, label: "Sensor (Optional)", short: "Sensor" },
  { id: 5, label: "AI Analysis", short: "AI" },
  { id: 6, label: "Results & Report", short: "Report" },
];

export default function ProgressStepper({ currentStep }: ProgressStepperProps) {
  return (
    <div className="bg-[var(--bg-card)]/80 backdrop-blur-xl border border-[var(--border-card)] rounded-2xl p-4 sm:p-5 text-xs font-semibold shadow-xl">
      {/* Mobile: current step summary + compact track */}
      <div className="sm:hidden space-y-3">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[var(--text-muted)] text-[11px] uppercase tracking-wider font-mono">
            STEP {Math.min(currentStep, STEPS.length)} / {STEPS.length}
          </span>
          <span className="text-[var(--text-main)] font-bold text-xs truncate flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-blue-500 animate-pulse" />
            {STEPS.find((s) => s.id === currentStep)?.label || STEPS[STEPS.length - 1].label}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          {STEPS.map((step) => {
            const isCompleted = step.id < currentStep;
            const isCurrent = step.id === currentStep;
            return (
              <div
                key={step.id}
                className={`h-2 flex-1 rounded-full transition-all duration-300 ${
                  isCompleted
                    ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.4)]"
                    : isCurrent
                    ? "bg-gradient-to-r from-blue-600 to-cyan-500 shadow-[0_0_10px_rgba(59,130,246,0.5)]"
                    : "bg-[var(--bg-card-sub)]"
                }`}
                title={step.label}
              />
            );
          })}
        </div>
      </div>

      {/* Desktop / tablet horizontal stepper */}
      <div className="hidden sm:flex items-center justify-between overflow-x-auto gap-2">
        {STEPS.map((step, index) => {
          const isCompleted = step.id < currentStep;
          const isCurrent = step.id === currentStep;

          return (
            <div key={step.id} className="flex items-center space-x-2.5 flex-shrink-0">
              <div
                className={`h-7 w-7 rounded-xl flex items-center justify-center text-xs font-bold transition-all duration-300 ${
                  isCompleted
                    ? "bg-emerald-500 text-slate-950 shadow-md shadow-emerald-500/25"
                    : isCurrent
                    ? "bg-gradient-to-br from-blue-600 to-indigo-600 text-white ring-4 ring-blue-500/25 shadow-lg shadow-blue-600/30 scale-105"
                    : "bg-[var(--bg-card-sub)] text-[var(--text-muted)] border border-[var(--border-card)]"
                }`}
              >
                {isCompleted ? <Check className="h-3.5 w-3.5 stroke-[3]" /> : step.id}
              </div>

              <span
                className={`text-xs ${
                  isCompleted
                    ? "text-[var(--text-sub)] font-medium"
                    : isCurrent
                    ? "text-[var(--text-main)] font-extrabold"
                    : "text-[var(--text-muted)] font-medium"
                }`}
              >
                <span className="md:hidden">{step.short}</span>
                <span className="hidden md:inline">{step.label}</span>
              </span>

              {index < STEPS.length - 1 && (
                <div
                  className={`h-[2px] w-6 lg:w-10 mx-1 lg:mx-2 rounded-full transition-all duration-300 ${
                    step.id < currentStep ? "bg-gradient-to-r from-emerald-500 to-emerald-400 shadow-[0_0_6px_rgba(16,185,129,0.3)]" : "bg-[var(--border-card)]"
                  }`}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

