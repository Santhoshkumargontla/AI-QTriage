"use client";

import { Sun, Moon, User } from "lucide-react";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

interface TopbarProps {
  title?: string;
}

function titleForPath(pathname: string | null): string {
  if (!pathname) return "AI-QTriage";
  if (pathname === "/") return "Dashboard";
  if (pathname.startsWith("/create-case")) return "New Assessment";
  if (pathname.startsWith("/cases/")) return "Case Detail";
  if (pathname.startsWith("/cases")) return "My Cases";
  if (pathname.startsWith("/research")) return "Research Results";
  return "AI-QTriage";
}

function applyTheme(newTheme: "dark" | "light") {
  document.documentElement.setAttribute("data-theme", newTheme);
  document.documentElement.classList.toggle("light", newTheme === "light");
  document.documentElement.classList.toggle("dark", newTheme === "dark");
  localStorage.setItem("aiqt-theme", newTheme);
}

export default function Topbar({ title }: TopbarProps) {
  const pathname = usePathname();
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const resolvedTitle = title ?? titleForPath(pathname);

  useEffect(() => {
    const stored = localStorage.getItem("aiqt-theme");
    if (stored === "light" || stored === "dark") {
      setTheme(stored);
      applyTheme(stored);
    }
  }, []);

  const changeTheme = (newTheme: "dark" | "light") => {
    setTheme(newTheme);
    applyTheme(newTheme);
  };

  return (
    <header className="h-16 bg-[var(--bg-topbar)]/80 backdrop-blur-xl border-b border-[var(--border-card)] px-6 flex items-center justify-between sticky top-0 z-30 transition-colors shadow-sm">
      <div className="flex items-center gap-3">
        <h2 className="text-xl font-extrabold text-[var(--text-main)] tracking-tight">
          {resolvedTitle}
        </h2>
        <span className="hidden sm:inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-emerald-950/80 text-emerald-400 border border-emerald-800/50 shadow-sm">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-ping" />
          SYSTEM OPERATIONAL
        </span>
      </div>

      <div className="flex items-center space-x-3.5">
        <div className="flex items-center bg-[var(--bg-card-sub)] p-1 rounded-2xl border border-[var(--border-card)] text-xs font-semibold shadow-inner">
          <button
            type="button"
            onClick={() => changeTheme("light")}
            className={`flex items-center space-x-1.5 px-3 py-1 rounded-xl transition-all ${
              theme === "light"
                ? "bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-bold shadow-md shadow-blue-600/30"
                : "text-[var(--text-muted)] hover:text-[var(--text-main)]"
            }`}
          >
            <Sun className="h-3.5 w-3.5" />
            <span>Light</span>
          </button>
          <button
            type="button"
            onClick={() => changeTheme("dark")}
            className={`flex items-center space-x-1.5 px-3 py-1 rounded-xl transition-all ${
              theme === "dark"
                ? "bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-bold shadow-md shadow-blue-600/30"
                : "text-[var(--text-muted)] hover:text-[var(--text-main)]"
            }`}
          >
            <Moon className="h-3.5 w-3.5" />
            <span>Dark</span>
          </button>
        </div>

        <div className="flex items-center space-x-2.5 bg-[var(--bg-card-sub)] pl-2 pr-3.5 py-1 rounded-2xl border border-[var(--border-card)] shadow-inner">
          <div className="h-7 w-7 rounded-xl bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center text-white font-bold text-xs shadow-md shadow-blue-500/20">
            <User className="h-3.5 w-3.5" />
          </div>
          <div className="text-left text-xs">
            <span className="font-extrabold text-[var(--text-main)] block leading-tight">Clinical Researcher</span>
            <span className="text-[10px] text-[var(--text-muted)] font-mono">DEMO-SESSION</span>
          </div>
        </div>
      </div>
    </header>
  );
}

