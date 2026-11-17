"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import Assistant from "@/components/Assistant";
import * as I from "@/components/Icons";
import { PAGES, useSession } from "@/lib/session";
import { useWorkspace } from "@/lib/workspace";

const ICONS: Record<string, (p: { size?: number }) => React.ReactElement> = {
  home: I.Home, briefcase: I.Briefcase, file: I.FileText, pen: I.Pen, layers: I.Layers,
  compass: I.Compass, users: I.Users, shield: I.Shield,
};

export default function AppShell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const { user, loading, signOut } = useSession();
  const { report } = useWorkspace();
  const [assistantOpen, setAssistantOpen] = useState(true);
  const [navOpen, setNavOpen] = useState(false);

  // On a phone the assistant starts closed, so the page isn't hidden behind it.
  useEffect(() => {
    if (window.matchMedia("(max-width: 900px)").matches) setAssistantOpen(false);
  }, []);
  useEffect(() => { setNavOpen(false); }, [path]);

  if (path === "/login") return <main className="auth-main">{children}</main>;
  if (loading || !user) return <main className="auth-main"><p className="muted small">Loading</p></main>;

  const pages = PAGES.filter((p) => p.roles.includes(user.role));
  const groups = [...new Set(pages.map((p) => p.group))];

  return (
    <div className={`app${assistantOpen ? " with-assistant" : ""}`}>
      <aside className={`sidebar${navOpen ? " open" : ""}`} aria-label="Main">
        <Link href="/" className="brand">
          <span className="brand-mark" aria-hidden="true">R</span>
          <span>ResumeMatch</span>
        </Link>
        <nav>
          {groups.map((g) => (
            <div key={g} className="nav-group">
              <p className="nav-label">{g}</p>
              {pages.filter((p) => p.group === g).map((p) => {
                const Icon = ICONS[p.icon];
                const locked = p.needsReport && !report;
                return (
                  <Link key={p.href} href={p.href} className={`nav-item${path === p.href ? " active" : ""}${locked ? " locked" : ""}`}
                        aria-current={path === p.href ? "page" : undefined}
                        title={locked ? "Upload your resume first" : undefined}>
                    <Icon size={17} />
                    <span>{p.label}</span>
                  </Link>
                );
              })}
            </div>
          ))}
        </nav>
        <div className="sidebar-foot">
          <div className="who-card">
            <span className="avatar" aria-hidden="true">{user.name.split(" ").map((w) => w[0]).join("").slice(0, 2)}</span>
            <span className="who-text"><b>{user.name}</b><span>{user.role_label}</span></span>
          </div>
          <button type="button" className="nav-item as-button" onClick={signOut}><I.Logout size={17} /><span>Sign out</span></button>
          <p className="attribution">Postings: LinkedIn Job Postings 2023 to 2024 by Arsh Koneru, CC BY-SA 4.0.</p>
        </div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <button type="button" className="icon-button only-mobile" onClick={() => setNavOpen(!navOpen)} aria-label="Menu" aria-expanded={navOpen}>
            <I.Menu />
          </button>
          <span className="topbar-title">{pages.find((p) => p.href === path)?.label ?? ""}</span>
          <span className="topbar-spacer" />
          {report && <span className="file-pill" title="The resume in this session">{"Resume checked"}</span>}
          <button type="button" className={`assistant-toggle${assistantOpen ? " on" : ""}`} onClick={() => setAssistantOpen(!assistantOpen)}
                  aria-expanded={assistantOpen} aria-controls="assistant">
            <I.Chat size={16} /><span>Assistant</span>
          </button>
        </header>
        <main className="page">{children}</main>
      </div>

      <Assistant open={assistantOpen} onClose={() => setAssistantOpen(false)} />
      {navOpen && <button type="button" className="scrim only-mobile" aria-label="Close menu" onClick={() => setNavOpen(false)} />}
    </div>
  );
}
