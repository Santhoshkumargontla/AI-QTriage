"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import {
  LayoutDashboard,
  PlusCircle,
  FolderKanban,
  FlaskConical,
  Lock,
  Menu,
  X,
  Sun,
  Moon,
  User,
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

function SidebarNav({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();
  return (
    <>
      <div className="p-4 sm:p-5 flex items-center space-x-3 border-b border-[var(--border-card)]">
        <div className="h-10 w-10 rounded-xl bg-blue-600/15 border border-blue-500/30 flex items-center justify-center text-blue-600 font-bold text-lg shrink-0">
          <FlaskConical className="h-5 w-5" />
        </div>
        <div className="min-w-0">
          <h1 className="font-extrabold text-[var(--text-main)] text-base sm:text-lg tracking-wide truncate">
            AI-QTriage
          </h1>
          <p className="text-xs sm:text-sm text-[var(--text-muted)] font-medium truncate">
            AI-Powered Injury Assessment
          </p>
        </div>
      </div>

      <nav className="p-3 space-y-1 mt-2 flex-1 overflow-y-auto">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = navIsActive(pathname, item.href);
          return (
            <Link
              key={item.name}
              href={item.href}
              onClick={onNavigate}
              className={`flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-sm font-semibold transition-all ${
                isActive
                  ? "bg-blue-600 text-white shadow-md shadow-blue-600/20"
                  : "text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-[var(--bg-card-sub)]"
              }`}
            >
              <Icon className={`h-4 w-4 shrink-0 ${isActive ? "text-white" : ""}`} />
              <span>{item.name}</span>
            </Link>
          );
        })}
      </nav>

      <div className="p-4 m-3 bg-[var(--bg-card-sub)] border border-[var(--border-card)] rounded-2xl space-y-2 text-sm">
        <span className="font-bold text-[var(--text-main)] block uppercase tracking-wider text-xs">
          AI-QTriage Research
        </span>
        <p className="text-[var(--text-muted)] leading-relaxed text-xs sm:text-sm">
          Research prototype for academic use only. Not for real medical diagnosis.
        </p>
        <div className="pt-2 border-t border-[var(--border-card)] flex items-center space-x-1.5 text-blue-600 font-semibold text-xs">
          <Lock className="h-3.5 w-3.5 shrink-0" />
          <span>Local & Environment Secured</span>
        </div>
      </div>
    </>
  );
}

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [mobileOpen, setMobileOpen] = useState(false);
  const resolvedTitle = titleForPath(pathname);

  useEffect(() => {
    const stored = localStorage.getItem("aiqt-theme");
    if (stored === "light" || stored === "dark") {
      setTheme(stored);
      applyTheme(stored);
    }
  }, []);

  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  useEffect(() => {
    if (!mobileOpen) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [mobileOpen]);

  const changeTheme = (newTheme: "dark" | "light") => {
    setTheme(newTheme);
    applyTheme(newTheme);
  };

  return (
    <div className="min-h-full flex font-sans bg-[var(--bg-main)] text-[var(--text-main)]">
      {/* Desktop sidebar */}
      <aside className="hidden md:flex w-64 bg-[var(--bg-sidebar)] border-r border-[var(--border-card)] flex-col justify-between h-screen sticky top-0 flex-shrink-0 z-40">
        <SidebarNav />
      </aside>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="md:hidden fixed inset-0 z-50 flex">
          <button
            type="button"
            aria-label="Close navigation"
            className="absolute inset-0 bg-black/50"
            onClick={() => setMobileOpen(false)}
          />
          <aside className="relative z-10 w-[min(18rem,88vw)] max-w-full h-full bg-[var(--bg-sidebar)] border-r border-[var(--border-card)] flex flex-col shadow-xl">
            <div className="flex justify-end p-2 border-b border-[var(--border-card)]">
              <button
                type="button"
                onClick={() => setMobileOpen(false)}
                className="p-2 rounded-lg text-[var(--text-muted)] hover:bg-[var(--bg-card-sub)]"
                aria-label="Close menu"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <SidebarNav onNavigate={() => setMobileOpen(false)} />
          </aside>
        </div>
      )}

      <div className="flex-1 flex flex-col min-w-0 w-full">
        <header className="min-h-14 sm:min-h-16 bg-[var(--bg-topbar)] border-b border-[var(--border-card)] px-3 sm:px-6 flex items-center justify-between sticky top-0 z-30 gap-2 py-2" style={{ paddingTop: "max(0.5rem, env(safe-area-inset-top))" }}>
          <div className="flex items-center gap-2 min-w-0 flex-1">
            <button
              type="button"
              className="md:hidden p-2 rounded-lg border border-[var(--border-card)] bg-[var(--bg-card-sub)] text-[var(--text-main)] shrink-0"
              onClick={() => setMobileOpen(true)}
              aria-label="Open navigation"
            >
              <Menu className="h-5 w-5" />
            </button>
            <h2 className="text-base sm:text-xl font-bold text-[var(--text-main)] tracking-tight truncate">
              {resolvedTitle}
            </h2>
          </div>

          <div className="flex items-center gap-1.5 sm:gap-4 shrink-0">
            <div className="flex items-center bg-[var(--bg-card-sub)] p-0.5 sm:p-1 rounded-xl border border-[var(--border-card)] text-xs sm:text-sm font-semibold">
              <button
                type="button"
                onClick={() => changeTheme("light")}
                className={`flex items-center gap-1 px-2 py-1.5 sm:py-1 rounded-lg transition-colors ${
                  theme === "light"
                    ? "bg-blue-600 text-white font-bold"
                    : "text-[var(--text-muted)] hover:text-[var(--text-main)]"
                }`}
                aria-label="Light theme"
              >
                <Sun className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
                <span className="hidden sm:inline">Light</span>
              </button>
              <button
                type="button"
                onClick={() => changeTheme("dark")}
                className={`flex items-center gap-1 px-2 py-1.5 sm:py-1 rounded-lg transition-colors ${
                  theme === "dark"
                    ? "bg-blue-600 text-white font-bold"
                    : "text-[var(--text-muted)] hover:text-[var(--text-main)]"
                }`}
                aria-label="Dark theme"
              >
                <Moon className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
                <span className="hidden sm:inline">Dark</span>
              </button>
            </div>

            <div className="hidden sm:flex items-center space-x-2.5 bg-[var(--bg-card-sub)] pl-2 pr-3 py-1 rounded-xl border border-[var(--border-card)]">
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

        <main className="flex-1 p-3 sm:p-4 md:p-6 max-w-[1600px] w-full mx-auto space-y-4 sm:space-y-6 overflow-x-hidden pb-[max(1rem,env(safe-area-inset-bottom))]">
          {children}
        </main>

        <footer className="border-t border-[var(--border-card)] bg-[var(--bg-main)] py-4 sm:py-5 px-3 sm:px-6 text-center text-xs sm:text-sm text-[var(--text-muted)] pb-[max(1rem,env(safe-area-inset-bottom))]">
          <div className="max-w-3xl mx-auto space-y-2 leading-relaxed">
            <p className="text-[var(--text-main)] font-semibold">
              Developed by Gontla Santhosh Kumar, Khamban Sai Girish, Dummu Rajesh
            </p>
            <p>
              Guided by Dr. Subba Lakshmi
            </p>
            <p className="pt-1 border-t border-[var(--border-card)] text-[11px] sm:text-xs">
              AI-QTriage is a research prototype for academic evaluation. It is not a validated medical device and must not be used for clinical diagnosis.
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
}
