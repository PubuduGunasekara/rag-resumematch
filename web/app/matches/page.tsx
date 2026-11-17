"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { Checklist, Facts, FitRing, NeedsReport } from "@/components/ReportParts";
import RoleGate from "@/components/RoleGate";
import type { Job } from "@/lib/types";
import { useWorkspace } from "@/lib/workspace";

const TIER: Record<Job["tier"], string> = { best: "Closest match", potential: "Worth a look", no: "Not a match yet" };

function MatchesView() {
  const { report } = useWorkspace();
  const params = useSearchParams();
  const router = useRouter();
  if (!report) return <NeedsReport what="Your matches list" />;
  const selected = report.jobs.find((j) => j.job_id === params.get("job")) ?? report.jobs.find((j) => j.job_id === report.best_job_id) ?? report.jobs[0];

  return (
    <div className="split">
      <section className="split-list" aria-label="Postings">
        <p className="eyebrow">{report.jobs.length} postings closest to your resume</p>
        {report.jobs.map((j) => (
          <button type="button" key={j.job_id} className={`match-card${j.job_id === selected?.job_id ? " active" : ""}`}
                  onClick={() => router.replace(`/matches?job=${j.job_id}`, { scroll: false })} aria-pressed={j.job_id === selected?.job_id}>
            <FitRing job={j} size={48} />
            <span className="match-card-text">
              <b>{j.title}</b>
              <span>{j.company}</span>
              <span className={`tier ${j.tier}`}>{TIER[j.tier]}</span>
            </span>
          </button>
        ))}
      </section>

      {selected && (
        <section className="split-detail card" aria-label={selected.title}>
          <div className="detail-head">
            <FitRing job={selected} size={84} />
            <div>
              <p className="eyebrow">{TIER[selected.tier]}</p>
              <h1 className="detail-title">{selected.title}</h1>
              <p className="muted">{selected.company}</p>
              <Facts details={selected.details} />
            </div>
          </div>
          <p className={`reason${selected.tier === "no" ? " warn" : ""}`}>{selected.reason}</p>
          <div className="btn-row">
            <Link className="btn primary" href={`/tailor?job=${selected.job_id}&tab=bullets`}>Suggest resume edits</Link>
            <Link className="btn ghost" href={`/tailor?job=${selected.job_id}&tab=letter`}>Cover letter</Link>
            <Link className="btn ghost" href={`/tailor?job=${selected.job_id}&tab=interview`}>Interview prep</Link>
            <Link className="btn ghost" href={`/resume?job=${selected.job_id}`}>See it in my resume</Link>
          </div>
          <h2 className="section-title">Requirements</h2>
          <Checklist job={selected} />
        </section>
      )}
    </div>
  );
}

export default function MatchesPage() {
  return <RoleGate href="/matches"><Suspense><MatchesView /></Suspense></RoleGate>;
}
