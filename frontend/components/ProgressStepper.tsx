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
    <div className="bg-[var(--bg-card)] border border-[var(--border-card)] rounded-2xl p-3 sm:p-4 text-sm font-semibold">
      {/* Mobile: current step summary + compact track */}
      <div className="sm:hidden space-y-3">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[var(--text-muted)] text-xs uppercase tracking-wide">
            Step {Math.min(currentStep, STEPS.length)} of {STEPS.length}
          </span>
          <span className="text-[var(--text-main)] font-bold text-sm truncate">
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
                className={`h-2 flex-1 rounded-full ${
                  isCompleted
                    ? "bg-emerald-500"
                    : isCurrent
                    ? "bg-blue-600"
                    : "bg-[var(--bg-card-sub)]"
                }`}
                title={step.label}
              />
            );
          })}
        </div>
      </div>

      {/* Desktop / tablet horizontal stepper */}
      <div className="hidden sm:flex items-center justify-between overflow-x-auto gap-1">
        {STEPS.map((step, index) => {
          const isCompleted = step.id < currentStep;
          const isCurrent = step.id === currentStep;

          return (
            <div key={step.id} className="flex items-center space-x-2 flex-shrink-0">
              <div
                className={`h-6 w-6 rounded-full flex items-center justify-center text-xs font-bold transition-all ${
                  isCompleted
                    ? "bg-emerald-500 text-slate-950"
                    : isCurrent
                    ? "bg-blue-600 text-white ring-2 ring-blue-500/50"
                    : "bg-[var(--bg-card-sub)] text-[var(--text-muted)]"
                }`}
              >
                {isCompleted ? <Check className="h-3.5 w-3.5 stroke-[3]" /> : step.id}
              </div>

              <span
                className={`text-sm ${
                  isCompleted
                    ? "text-[var(--text-sub)]"
                    : isCurrent
                    ? "text-[var(--text-main)] font-bold"
                    : "text-[var(--text-muted)]"
                }`}
              >
                <span className="md:hidden">{step.short}</span>
                <span className="hidden md:inline">{step.label}</span>
              </span>

              {index < STEPS.length - 1 && (
                <div
                  className={`h-[1px] w-6 lg:w-8 mx-1 lg:mx-2 ${
                    step.id < currentStep ? "bg-emerald-500/60" : "bg-[var(--border-card)]"
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
