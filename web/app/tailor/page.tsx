"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import * as I from "@/components/Icons";
import { NeedsReport } from "@/components/ReportParts";
import RoleGate from "@/components/RoleGate";
import { BulletsView, InterviewView, LetterView, type Bullets, type Interview, type Letter } from "@/components/Tailor";
import { useWorkspace, type TailorKind } from "@/lib/workspace";

const TABS: { kind: TailorKind; label: string; icon: (p: { size?: number }) => React.ReactElement; intro: string; button: string }[] = [
  { kind: "bullets", label: "Resume edits", icon: I.Pen, button: "Suggest edits",
    intro: "Rewrites of the bullets closest to this posting, in its wording. A rewrite that adds a number, a tool or a word your resume doesn't use is removed, with the reason." },
  { kind: "letter", label: "Cover letter", icon: I.FileText, button: "Draft the letter",
    intro: "A short letter built from the requirements your resume shows and your own lines. Tools, names and numbers are checked against your resume; underlined sentences come from it." },
  { kind: "interview", label: "Interview prep", icon: I.Chat, button: "Prepare questions",
    intro: "The question an interviewer is likely to ask about each requirement, gaps first. What to draw on comes from your resume, never from the model." },
];

function TailorView() {
  const ws = useWorkspace();
  const params = useSearchParams();
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const report = ws.report;
  if (!report) return <NeedsReport what="Tailoring" />;
  const job = report.jobs.find((j) => j.job_id === params.get("job")) ?? report.jobs.find((j) => j.job_id === report.best_job_id) ?? report.jobs[0];
  const tab = TABS.find((t) => t.kind === params.get("tab")) ?? TABS[0];
  const result = ws.tailored[job.job_id]?.[tab.kind];
  const running = ws.tailoring?.job_id === job.job_id && ws.tailoring.kind === tab.kind;
  const go = (jobId: string, kind: TailorKind) => router.replace(`/tailor?job=${jobId}&tab=${kind}`, { scroll: false });

  async function run() {
    setError(null);
    const r = await ws.runTailor(job.job_id, tab.kind);
    if (!r.ok && r.message) setError(r.message);
  }

  return (
    <div className="tailor-page">
      <section className="hero compact">
        <p className="eyebrow">Tailor for a posting</p>
        <label className="select-wrap big">
          <span className="sr-only">Posting</span>
          <select value={job.job_id} onChange={(e) => go(e.target.value, tab.kind)}>
            {report.jobs.map((j) => <option key={j.job_id} value={j.job_id}>{j.title}{j.company ? `, ${j.company}` : ""}</option>)}
          </select>
        </label>
        <p className="muted small">{job.met} of {job.checked} requirements shown on your resume. Results stay in this tab while you work.</p>
      </section>

      <div className="tabs" role="tablist">
        {TABS.map((t) => {
          const Icon = t.icon;
          const done = !!ws.tailored[job.job_id]?.[t.kind];
          return (
            <button key={t.kind} role="tab" type="button" aria-selected={t.kind === tab.kind} className={`tab${t.kind === tab.kind ? " active" : ""}`}
                    onClick={() => go(job.job_id, t.kind)}>
              <Icon size={16} /> {t.label} {done && <span className="tab-dot" aria-label="ready" />}
            </button>
          );
        })}
      </div>

      <section className="card tab-panel" role="tabpanel">
        <p className="tab-intro">{tab.intro}</p>
        <div className="btn-row">
          <button type="button" className="btn primary" disabled={!!ws.tailoring} onClick={run}>
            {running ? "Working, usually 10 to 15 s" : result ? "Try again" : tab.button}
          </button>
        </div>
        {running && <div className="skeleton" aria-hidden="true"><i /><i /><i /></div>}
        {error && <p className="banner error" role="alert">{error}</p>}
        {!running && result && tab.kind === "bullets" ? <BulletsView data={result as Bullets} /> : null}
        {!running && result && tab.kind === "letter" ? <LetterView data={result as Letter} /> : null}
        {!running && result && tab.kind === "interview" ? <InterviewView data={{ ...(result as Interview), kind: "interview" }} /> : null}
      </section>
    </div>
  );
}

export default function TailorPage() {
  return <RoleGate href="/tailor"><Suspense><TailorView /></Suspense></RoleGate>;
}
