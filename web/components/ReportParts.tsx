"use client";

import * as I from "@/components/Icons";
import type { Job, Requirement } from "@/lib/types";

export const STATUS_LABEL: Record<Requirement["status"], string> = { shown: "Shown", partly: "Partly shown", not_found: "Not found" };

/** met / checked as a ring. Not a score: it counts requirements the resume shows. */
export function FitRing({ job, size = 76 }: { job: Job; size?: number }) {
  const r = (size - 8) / 2;
  const c = 2 * Math.PI * r;
  const share = job.checked ? job.met / job.checked : 0;
  return (
    <div className="ring" style={{ width: size, height: size }} role="img"
         aria-label={job.checked ? `${job.met} of ${job.checked} requirements shown` : "No requirements checked"}>
      <svg width={size} height={size} aria-hidden="true">
        <circle cx={size / 2} cy={size / 2} r={r} className="ring-track" />
        <circle cx={size / 2} cy={size / 2} r={r} className="ring-fill" strokeDasharray={`${c * share} ${c}`}
                transform={`rotate(-90 ${size / 2} ${size / 2})`} />
      </svg>
      <span className="ring-text">
        <b>{job.checked ? job.met : "-"}</b>
        {size >= 60 ? <small>{job.checked ? `of ${job.checked}` : "not checked"}</small> : job.checked ? <small>of {job.checked}</small> : null}
      </span>
    </div>
  );
}

export function Facts({ details }: { details: Job["details"] }) {
  const items = [details.location, details.work_type, details.level, details.remote ? "Remote allowed" : null, details.salary]
    .filter(Boolean) as string[];
  if (!items.length) return null;
  return <div className="fact-pills">{items.map((i) => <span key={i}>{i}</span>)}</div>;
}

export function StatusIcon({ status }: { status: Requirement["status"] }) {
  if (status === "shown") return <span className="st st-shown" title="Shown on your resume"><I.Check size={14} /></span>;
  if (status === "partly") return <span className="st st-partly" title="Partly shown"><I.Half size={14} /></span>;
  return <span className="st st-missing" title="Not found on your resume"><I.Dash size={14} /></span>;
}

/** A requirement as a pill: filled when shown, half when partly, dashed when not found. */
export function RequirementChip({ req, active, onClick }: { req: Requirement; active?: boolean; onClick?: () => void }) {
  return (
    <button type="button" className={`req-chip ${req.status}${active ? " active" : ""}`} onClick={onClick}
            aria-pressed={active} title={req.resume_quote ? `Your resume: "${req.resume_quote}"` : STATUS_LABEL[req.status]}>
      <StatusIcon status={req.status} />
      <span>{req.text}</span>
      {req.importance === "required" && <em>required</em>}
    </button>
  );
}

export function Checklist({ job }: { job: Job }) {
  const counted = job.requirements.filter((r) => !r.soft);
  const soft = job.requirements.filter((r) => r.soft).map((r) => r.text);
  return (
    <div className="check-list">
      {counted.length === 0 && (
        <p className="muted small">No requirements in this posting could be checked against your resume. Open the posting to read it yourself.</p>
      )}
      {counted.map((r, i) => (
        <div className={`check-row ${r.status}`} key={i}>
          <StatusIcon status={r.status} />
          <div className="check-body">
            <p className="check-title">
              {r.text}
              {r.importance !== "unspecified" && <span className={`imp ${r.importance}`}>{r.importance}</span>}
              {r.credential && <span className="imp credential">credential</span>}
            </p>
            {r.met && r.resume_quote && <p className="check-quote">&ldquo;{r.resume_quote}&rdquo;</p>}
            {!r.met && r.partly.length > 0 && <p className="check-quote">Your resume mentions {r.partly.join(", ")}, not the rest.</p>}
          </div>
          <span className={`check-status ${r.status}`}>{STATUS_LABEL[r.status]}</span>
        </div>
      ))}
      {soft.length > 0 && (
        <p className="soft-note">Also asks for general strengths: {soft.join("; ")}. Your interview and cover letter are the place for these.</p>
      )}
      {job.url && (
        <a className="posting-link" href={job.url} target="_blank" rel="noopener noreferrer">
          Open the original posting <I.External size={13} /> <span className="muted">(it may have closed)</span>
        </a>
      )}
    </div>
  );
}

/** "Upload your resume first" for pages that need a report. */
export function NeedsReport({ what }: { what: string }) {
  return (
    <div className="empty-state">
      <span className="empty-icon"><I.Upload size={22} /></span>
      <h2>Upload your resume first</h2>
      <p>{what} comes from your resume check. Start there, or ask the assistant what to expect.</p>
      <a className="btn primary" href="/">Go to Start</a>
    </div>
  );
}
