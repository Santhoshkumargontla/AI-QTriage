"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  PlusCircle,
  FolderKanban,
  FlaskConical,
  Lock
} from "lucide-react";

const NAV_ITEMS = [
  { name: "Dashboard", href: "/", icon: LayoutDashboard },
  { name: "New Assessment", href: "/create-case", icon: PlusCircle },
  { name: "My Cases", href: "/cases", icon: FolderKanban },
  { name: "Research Results", href: "/research", icon: FlaskConical },
];

function navIsActive(pathname: string | null, href: string): boolean {
  if (!pathname) return false;
  if (href === "/") return pathname === "/";
  return pathname === href || pathname.startsWith(`${href}/`) || pathname.startsWith(`${href}?`);
}

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 bg-[var(--bg-sidebar)]/90 backdrop-blur-xl border-r border-[var(--border-card)] flex flex-col justify-between h-screen sticky top-0 flex-shrink-0 z-40 shadow-2xl">
      <div>
        {/* Brand Header */}
        <div className="p-5 flex items-center space-x-3.5 border-b border-[var(--border-card)]">
          <div className="relative h-11 w-11 rounded-2xl bg-gradient-to-br from-blue-600 via-indigo-600 to-cyan-500 p-0.5 shadow-lg shadow-blue-500/25">
            <div className="h-full w-full bg-slate-950 rounded-[14px] flex items-center justify-center text-cyan-400">
              <FlaskConical className="h-5 w-5 animate-pulse" />
            </div>
          </div>
          <div>
            <h1 className="font-extrabold text-[var(--text-main)] text-lg tracking-wide flex items-center gap-1.5">
              <span>AI-QTriage</span>
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-ping"></span>
            </h1>
            <p className="text-xs text-[var(--text-muted)] font-medium">
              Medical Decision System
            </p>
          </div>
        </div>

        {/* Navigation Items */}
        <nav className="p-3 space-y-1.5 mt-3">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = navIsActive(pathname, item.href);

            return (
              <Link
                key={item.name}
                href={item.href}
                className={`relative flex items-center space-x-3 px-4 py-3 rounded-2xl text-sm font-semibold transition-all duration-200 group ${
                  isActive
                    ? "bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg shadow-blue-600/30 scale-[1.02]"
                    : "text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-[var(--bg-card-sub)] hover:scale-[1.01]"
                }`}
              >
                <Icon className={`h-4.5 w-4.5 transition-transform group-hover:scale-110 ${isActive ? "text-white" : "text-blue-400/80"}`} />
                <span>{item.name}</span>
                {isActive && (
                  <span className="absolute right-3 h-2 w-2 rounded-full bg-white animate-pulse" />
                )}
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Footer Info Card */}
      <div className="p-4 m-3.5 bg-[var(--bg-card-sub)] border border-[var(--border-card)] rounded-2xl space-y-2 text-xs backdrop-blur-md">
        <div className="flex items-center justify-between">
          <span className="font-bold text-[var(--text-main)] block uppercase tracking-wider text-[10px]">
            AI-QTriage Core v2.5
          </span>
          <span className="px-1.5 py-0.5 rounded text-[9px] font-mono font-bold bg-emerald-950 text-emerald-400 border border-emerald-800/40">
            ONLINE
          </span>
        </div>
        <p className="text-[var(--text-muted)] leading-relaxed text-[11px]">
          Research prototype for academic evaluation. Not for clinical diagnosis.
        </p>
        <div className="pt-2 border-t border-[var(--border-card)] flex items-center space-x-1.5 text-blue-400 font-semibold text-[11px]">
          <Lock className="h-3.5 w-3.5" />
          <span>Local Data Sandbox Secured</span>
        </div>
      </div>
    </aside>
  );
}

