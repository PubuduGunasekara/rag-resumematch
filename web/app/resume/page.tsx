"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Fragment, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { NeedsReport, RequirementChip } from "@/components/ReportParts";
import RoleGate from "@/components/RoleGate";
import type { Requirement } from "@/lib/types";
import { useWorkspace } from "@/lib/workspace";

/** The resume as written, with marks. The active span gets its own style and is scrolled to. */
function ResumeDoc({ text, spans, active }: { text: string; spans: [number, number][]; active: [number, number] | null }) {
  const ref = useRef<HTMLDivElement>(null);
  // Scroll the window only: scrollIntoView also moved the sticky app shell.
  useEffect(() => {
    const mark = ref.current?.querySelector("mark.active");
    if (!mark) return;
    const top = mark.getBoundingClientRect().top + window.scrollY - window.innerHeight / 3;
    window.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
  }, [active]);
  const all = [...spans];
  if (active && !all.some(([s, e]) => s === active[0] && e === active[1])) all.push(active);
  all.sort((a, b) => a[0] - b[0]);
  const merged: [number, number][] = [];
  for (const [s, e] of all) {
    const last = merged[merged.length - 1];
    if (last && s <= last[1]) last[1] = Math.max(last[1], e); else merged.push([s, e]);
  }
  const parts: React.ReactNode[] = [];
  let pos = 0;
  merged.forEach(([s, e], i) => {
    const isActive = !!active && s <= active[0] && e >= active[1];
    parts.push(<Fragment key={`t${i}`}>{text.slice(pos, s)}</Fragment>, <mark key={`m${i}`} className={isActive ? "active" : ""}>{text.slice(s, e)}</mark>);
    pos = e;
  });
  parts.push(<Fragment key="end">{text.slice(pos)}</Fragment>);
  return <div className="paper" ref={ref}>{parts}</div>;
}

function ResumeView() {
  const { report } = useWorkspace();
  const params = useSearchParams();
  const router = useRouter();
  const [activeReq, setActiveReq] = useState<string | null>(null);
  const job = useMemo(() => report && (report.jobs.find((j) => j.job_id === params.get("job"))
    ?? report.jobs.find((j) => j.job_id === report.best_job_id) ?? report.jobs[0]), [report, params]);
  useEffect(() => { setActiveReq(null); }, [job?.job_id]);
  if (!report || !job) return <NeedsReport what="The evidence view" />;

  const text = report.resume_text.trim();
  const offset = report.resume_text.indexOf(text);   // spans are on the untrimmed text
  const shift = (s: [number, number]): [number, number] => [s[0] - offset, s[1] - offset];
  const counted = job.requirements.filter((r) => !r.soft);
  const groups: { title: string; status: Requirement["status"]; hint: string }[] = [
    { title: "Shown on your resume", status: "shown", hint: "Click one to see the line that shows it." },
    { title: "Partly shown", status: "partly", hint: "Some of the words are on your resume, not all." },
    { title: "Not found", status: "not_found", hint: "Add it if it's true. Wording matters." },
  ];
  const active = counted.find((r) => r.text === activeReq);

  return (
    <div className="evidence">
      <section className="evidence-panel card">
        <p className="eyebrow">Evidence for</p>
        <label className="select-wrap">
          <span className="sr-only">Posting</span>
          <select value={job.job_id} onChange={(e) => router.replace(`/resume?job=${e.target.value}`, { scroll: false })}>
            {report.jobs.map((j) => <option key={j.job_id} value={j.job_id}>{j.title}{j.company ? `, ${j.company}` : ""}</option>)}
          </select>
        </label>
        <div className="evidence-stats">
          <span className="stat shown"><b>{counted.filter((r) => r.status === "shown").length}</b> shown</span>
          <span className="stat partly"><b>{counted.filter((r) => r.status === "partly").length}</b> partly</span>
          <span className="stat missing"><b>{counted.filter((r) => r.status === "not_found").length}</b> not found</span>
        </div>
        {counted.length === 0 && <p className="muted small">No requirements of this posting could be checked.</p>}
        {groups.map((g) => {
          const reqs = counted.filter((r) => r.status === g.status);
          if (!reqs.length) return null;
          return (
            <div className="chip-group" key={g.status}>
              <p className="chip-group-title">{g.title}</p>
              <p className="muted small">{g.hint}</p>
              <div className="chips-wrap">
                {reqs.map((r) => (
                  <RequirementChip key={r.text} req={r} active={activeReq === r.text}
                                   onClick={() => setActiveReq(activeReq === r.text ? null : r.text)} />
                ))}
              </div>
            </div>
          );
        })}
        {active && (
          <div className="focus-box">
            <p className="chip-group-title">{active.text}</p>
            {active.resume_quote ? <p>Your resume: &ldquo;{active.resume_quote}&rdquo;</p>
              : active.partly.length ? <p>Your resume mentions {active.partly.join(", ")}, not the rest.</p>
              : <p>Your resume doesn&apos;t use these words. If you have it under another name, consider saying it the way the posting does.</p>}
            {active.posting_sentence && <p className="muted small">The posting: &ldquo;{active.posting_sentence}&rdquo;</p>}
          </div>
        )}
      </section>

      <section className="evidence-doc">
        <div className="doc-bar">
          <span className="legend"><mark>marked</mark> lines show a requirement of this posting</span>
          {active?.spans.length ? <span className="legend"><mark className="active">selected</mark> {active.text}</span> : null}
        </div>
        <ResumeDoc text={text} spans={job.highlight.map(shift)} active={active?.spans[0] ? shift(active.spans[0]) : null} />
      </section>
    </div>
  );
}

export default function ResumePage() {
  return <RoleGate href="/resume"><Suspense><ResumeView /></Suspense></RoleGate>;
}
