"use client";

import { Fragment, useMemo, useState } from "react";
import type { Job, Report, Requirement } from "@/lib/types";

const STATUS: Record<Requirement["status"], string> = { shown: "Shown", partly: "Partly shown", not_found: "Not found" };

function Facts({ details }: { details: Job["details"] }) {
  const items = [details.location, details.work_type, details.level, details.remote ? "Remote allowed" : null, details.salary]
    .filter(Boolean) as string[];
  if (!items.length) return null;
  return <div className="facts">{items.map((i) => <span key={i}>{i}</span>)}</div>;
}

function Score({ job }: { job: Job }) {
  const counted = job.requirements.filter((r) => !r.soft);
  if (!counted.length) return <p className="small muted" style={{ margin: 0 }}>No specific requirements could be checked</p>;
  return (
    <div className="score">
      <span className="seg" aria-hidden="true">
        {counted.map((r, i) => <i key={i} className={r.status === "not_found" ? "" : r.status} />)}
      </span>
      <span>
        {job.met} of {job.checked} {job.checked === 1 ? "requirement" : "requirements"} shown on your resume{job.partly ? `, ${job.partly} partly` : ""}
      </span>
    </div>
  );
}

function Checklist({ job }: { job: Job }) {
  const counted = job.requirements.filter((r) => !r.soft);
  const soft = job.requirements.filter((r) => r.soft).map((r) => r.text);
  return (
    <div className="checklist">
      {counted.length === 0 && (
        <p className="small muted">
          The model found no requirements in this posting that could be checked against your resume. Open the posting
          to read it yourself.
        </p>
      )}
      {counted.map((r, i) => (
        <div className="req-row" key={i}>
          <div>
            {r.text}
            {r.importance !== "unspecified" && <span className="imp">{r.importance}</span>}
            {r.met && r.resume_quote && <span className="quote">Your resume: &ldquo;{r.resume_quote}&rdquo;</span>}
            {!r.met && r.partly.length > 0 && (
              <span className="quote">Your resume mentions {r.partly.join(", ")}, not the rest</span>
            )}
          </div>
          <span className={`status ${r.status}`}>{STATUS[r.status]}</span>
        </div>
      ))}
      {soft.length > 0 && (
        <p className="soft">
          It also asks for general strengths: {soft.join("; ")}. Your interview and cover letter are the place for these.
        </p>
      )}
      {job.url && (
        <p className="posting-link">
          <a href={job.url} target="_blank" rel="noopener noreferrer">Open the original posting</a>{" "}
          <span className="muted">(it may have closed)</span>
        </p>
      )}
    </div>
  );
}

function JobRow({ job, open, onToggle }: { job: Job; open: boolean; onToggle: () => void }) {
  const id = `job-${job.job_id}`;
  return (
    <div className="job">
      <button className="job-toggle" aria-expanded={open} aria-controls={id} onClick={onToggle}>
        <span>
          <span className="title">{job.title}</span>
          {job.company && <span className="company">{job.company}</span>}
        </span>
        <span className="count">{job.checked ? `${job.met} of ${job.checked} shown` : "Not checked"}</span>
      </button>
      {open && (
        <div className="job-body" id={id}>
          <Facts details={job.details} />
          <Score job={job} />
          <p className={`reason${job.tier === "no" ? " no" : ""}`}>{job.reason}</p>
          <Checklist job={job} />
        </div>
      )}
    </div>
  );
}

function Resume({ text, spans }: { text: string; spans: [number, number][] }) {
  const parts: React.ReactNode[] = [];
  let pos = 0;
  spans.forEach(([start, end], i) => {
    parts.push(<Fragment key={`t${i}`}>{text.slice(pos, start)}</Fragment>, <mark key={`m${i}`}>{text.slice(start, end)}</mark>);
    pos = end;
  });
  parts.push(<Fragment key="end">{text.slice(pos)}</Fragment>);
  return <div className="resume">{parts}</div>;
}

export default function ReportView({ report, fileName }: { report: Report; fileName: string }) {
  const best = report.jobs.find((j) => j.job_id === report.best_job_id) ?? null;
  const others = report.jobs.filter((j) => j.tier === "potential");
  const blocked = report.jobs.filter((j) => j.tier === "no");
  const [open, setOpen] = useState<string | null>(null);
  const withEvidence = useMemo(() => report.jobs.filter((j) => j.highlight.length > 0), [report]);
  const [evidenceFor, setEvidenceFor] = useState<string | null>(withEvidence[0]?.job_id ?? null);
  const shown = withEvidence.find((j) => j.job_id === evidenceFor);

  const toggle = (job: Job) => {
    setOpen(open === job.job_id ? null : job.job_id);
    if (job.highlight.length) setEvidenceFor(job.job_id);
  };

  return (
    <section className="report" aria-label={`Report for ${fileName}`}>
      <p className="tag">Resume report</p>
      <h1 className="serif">{report.headline}</h1>
      <p className="lede">
        {report.summary}
        {report.total_postings ? ` Searched ${report.total_postings.toLocaleString()} LinkedIn postings.` : ""}
      </p>

      {best && (
        <div className="section">
          <p className="tag">Closest match</p>
          <div className="lead">
            <h2 className="serif">{best.title}</h2>
            {best.company && <p className="muted" style={{ margin: "2px 0 0", fontSize: 15 }}>{best.company}</p>}
            <Facts details={best.details} />
            <Score job={best} />
            <p className="reason">{best.reason}</p>
            <Checklist job={best} />
          </div>
        </div>
      )}

      {others.length > 0 && (
        <div className="section">
          <p className="tag">{best ? "Also worth a look" : "The closest postings"}</p>
          {others.map((j) => <JobRow key={j.job_id} job={j} open={open === j.job_id} onToggle={() => toggle(j)} />)}
        </div>
      )}

      {blocked.length > 0 && (
        <div className="section">
          <p className="tag">Not a match right now</p>
          {blocked.map((j) => <JobRow key={j.job_id} job={j} open={open === j.job_id} onToggle={() => toggle(j)} />)}
        </div>
      )}

      <div className="section">
        <p className="tag">What to work on, across these postings</p>
        {report.learn_first.length === 0 ? (
          <p className="small muted">Nothing came up in more than one place. The checklists above are the best guide.</p>
        ) : (
          report.learn_first.map((item) => (
            <div className="learn" key={item.text}>
              <span>{item.text}{item.credential && <span className="imp muted small"> credential</span>}</span>
              <span>
                <div className="meter"><div style={{ width: `${(100 * item.jobs) / report.jobs.length}%` }} /></div>
                <span className="small muted">{item.jobs} of {report.jobs.length} postings</span>
              </span>
            </div>
          ))
        )}
      </div>

      <div className="section">
        <p className="tag">Your resume, with the evidence marked</p>
        {withEvidence.length > 0 && (
          <div className="picker" role="group" aria-label="Show evidence for">
            {withEvidence.map((j) => (
              <button key={j.job_id} aria-pressed={j.job_id === evidenceFor} onClick={() => setEvidenceFor(j.job_id)}>
                {j.title}{j.company ? `, ${j.company}` : ""}
              </button>
            ))}
          </div>
        )}
        <Resume text={report.resume_text.trim()} spans={shown?.highlight ?? []} />
      </div>

      <div className="section">
        <p className="tag">How this was checked</p>
        <details className="how">
          <summary>Search: {report.how.sections.length} resume sections, {report.how.retrieved.length} postings</summary>
          <p className="muted" style={{ margin: "10px 0 0" }}>
            Each section of your resume was searched separately, since the embedding model reads at most 512 tokens at
            a time, and results were combined per posting with reciprocal rank fusion. Sections: {report.how.sections.join(", ")}.
          </p>
          <ol>
            {report.how.retrieved.map((p, i) => (
              <li key={i}>{p.title}, {p.company ?? "unknown company"} (found by {p.matched_sections.slice(0, 3).join(", ")})</li>
            ))}
          </ol>
        </details>
        <details className="how">
          <summary>Typo fixes used for the search: {report.how.corrections.length}</summary>
          {report.how.corrections.length === 0 ? (
            <p className="muted" style={{ margin: "10px 0 0" }}>None. Your resume was searched exactly as written.</p>
          ) : (
            <ul>{report.how.corrections.map((c, i) => <li key={i}>{c.original} searched as {c.corrected}</li>)}</ul>
          )}
        </details>
        <details className="how">
          <summary>Model claims that failed a check: {report.how.rejected.length}</summary>
          <p className="muted" style={{ margin: "10px 0 0" }}>
            Every claim from the model is checked in code before it is shown. These failed and were left out.
          </p>
          <ul>{report.how.rejected.map((c, i) => <li key={i}>{c.requirement} (claimed {c.claimed_as}): {c.issue}</li>)}</ul>
        </details>
      </div>

      <p className="disclaimer">
        Checks compare wording, so a requirement marked not found may still be something you have. Add it to your resume
        if it&rsquo;s true.
      </p>
    </section>
  );
}
