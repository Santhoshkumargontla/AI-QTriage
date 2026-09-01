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
    <aside className="w-64 bg-[var(--bg-sidebar)] border-r border-[var(--border-card)] flex flex-col justify-between h-screen sticky top-0 flex-shrink-0 z-40">
      <div>
        <div className="p-5 flex items-center space-x-3 border-b border-[var(--border-card)]">
          <div className="h-10 w-10 rounded-xl bg-blue-600/15 border border-blue-500/30 flex items-center justify-center text-blue-600 font-bold text-lg">
            <FlaskConical className="h-5 w-5" />
          </div>
          <div>
            <h1 className="font-extrabold text-[var(--text-main)] text-lg tracking-wide flex items-center gap-1.5">
              AI-QTriage
            </h1>
            <p className="text-sm text-[var(--text-muted)] font-medium">
              AI-Powered Injury Assessment
            </p>
          </div>
        </div>

        <nav className="p-3 space-y-1 mt-2">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = navIsActive(pathname, item.href);

            return (
              <Link
                key={item.name}
                href={item.href}
                className={`flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-sm font-semibold transition-all ${
                  isActive
                    ? "bg-blue-600 text-white shadow-md shadow-blue-600/20"
                    : "text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-[var(--bg-card-sub)]"
                }`}
              >
                <Icon className={`h-4 w-4 ${isActive ? "text-white" : ""}`} />
                <span>{item.name}</span>
              </Link>
            );
          })}
        </nav>
      </div>

      <div className="p-4 m-3 bg-[var(--bg-card-sub)] border border-[var(--border-card)] rounded-2xl space-y-2 text-sm">
        <span className="font-bold text-[var(--text-main)] block uppercase tracking-wider">
          AI-QTriage Research
        </span>
        <p className="text-[var(--text-muted)] leading-relaxed">
          This is a research prototype for academic use only. Not for real medical diagnosis.
        </p>
        <div className="pt-2 border-t border-[var(--border-card)] flex items-center space-x-1.5 text-blue-600 font-semibold">
          <Lock className="h-3.5 w-3.5" />
          <span>Local & Environment Secured</span>
        </div>
      </div>
    </aside>
  );
}
