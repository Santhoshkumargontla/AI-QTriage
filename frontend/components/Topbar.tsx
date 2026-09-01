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
    <header className="h-16 bg-[var(--bg-topbar)] border-b border-[var(--border-card)] px-6 flex items-center justify-between sticky top-0 z-30 transition-colors">
      <div>
        <h2 className="text-xl font-bold text-[var(--text-main)] tracking-tight">
          {resolvedTitle}
        </h2>
      </div>

      <div className="flex items-center space-x-4">
        <div className="flex items-center bg-[var(--bg-card-sub)] p-1 rounded-xl border border-[var(--border-card)] text-sm font-semibold">
          <button
            type="button"
            onClick={() => changeTheme("light")}
            className={`flex items-center space-x-1 px-2.5 py-1 rounded-lg transition-colors ${
              theme === "light"
                ? "bg-blue-600 text-white font-bold"
                : "text-[var(--text-muted)] hover:text-[var(--text-main)]"
            }`}
          >
            <Sun className="h-4 w-4" />
            <span>Light</span>
          </button>
          <button
            type="button"
            onClick={() => changeTheme("dark")}
            className={`flex items-center space-x-1 px-2.5 py-1 rounded-lg transition-colors ${
              theme === "dark"
                ? "bg-blue-600 text-white font-bold"
                : "text-[var(--text-muted)] hover:text-[var(--text-main)]"
            }`}
          >
            <Moon className="h-4 w-4" />
            <span>Dark</span>
          </button>
        </div>

        <div className="flex items-center space-x-2.5 bg-[var(--bg-card-sub)] pl-2 pr-3 py-1 rounded-xl border border-[var(--border-card)]">
          <div className="h-7 w-7 rounded-lg bg-blue-600/15 border border-blue-500/30 flex items-center justify-center text-blue-600 font-bold text-xs">
            <User className="h-4 w-4" />
          </div>
          <div className="text-left text-sm">
            <span className="font-bold text-[var(--text-main)] block leading-tight">Researcher</span>
            <span className="text-xs text-[var(--text-muted)] font-mono">AIQT-2024</span>
          </div>
        </div>
      </div>
    </header>
  );
}
