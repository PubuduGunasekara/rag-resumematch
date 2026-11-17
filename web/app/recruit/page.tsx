"use client";

import { useRef, useState } from "react";
import RoleGate from "@/components/RoleGate";
import { handleSignedOut } from "@/lib/session";

// Mirrors resumematch.api.app.ranking_to_json.
type Count = { shown: number; total: number };
type Candidate = {
  label: string; identity: { name: string | null; file: string };
  counts: { required: Count; unspecified: Count; preferred: Count }; partly: number;
  checklist: { requirement: string; importance: string; soft: boolean; status: string; evidence: string | null }[];
};
type Ranking = {
  blind: boolean; note: string; failed: { file: string; message: string }[];
  requirements: { text: string; importance: string; soft: boolean; issue: string | null }[];
  candidates: Candidate[];
};

const STATUS: Record<string, string> = { shown: "Shown", partly: "Partly", not_found: "Not found" };
const IMPORTANCE: Record<string, string> = { required: "required", preferred: "preferred", unspecified: "" };

function Counts({ c }: { c: Candidate }) {
  const parts = [
    c.counts.required.total ? `Required ${c.counts.required.shown} of ${c.counts.required.total}` : null,
    c.counts.unspecified.total ? `Other ${c.counts.unspecified.shown} of ${c.counts.unspecified.total}` : null,
    c.counts.preferred.total ? `Preferred ${c.counts.preferred.shown} of ${c.counts.preferred.total}` : null,
    c.partly ? `${c.partly} partly` : null,
  ].filter(Boolean);
  return <span className="count">{parts.join(" · ")}</span>;
}

function Results({ data, reveal }: { data: Ranking; reveal: boolean }) {
  const [open, setOpen] = useState<string | null>(data.candidates[0]?.label ?? null);
  const dropped = data.requirements.filter((r) => r.issue);
  const soft = data.requirements.filter((r) => !r.issue && r.soft);
  return (
    <section className="report" aria-label="Ranked candidates">
      <p className="tag">Ranked candidates</p>
      {data.candidates.map((c) => (
        <div className="job" key={c.label}>
          <button className="job-toggle" aria-expanded={open === c.label} onClick={() => setOpen(open === c.label ? null : c.label)}>
            <span>
              <span className="title">{c.label}</span>
              {reveal && <span className="company">{c.identity.name ?? "Name not found"} ({c.identity.file})</span>}
            </span>
            <Counts c={c} />
          </button>
          {open === c.label && (
            <div className="job-body checklist">
              {c.checklist.filter((r) => !r.soft).map((r, i) => (
                <div className="req-row" key={i}>
                  <div>
                    {r.requirement}
                    {IMPORTANCE[r.importance] && <span className="imp">{IMPORTANCE[r.importance]}</span>}
                    {r.evidence && <span className="quote">Resume: &ldquo;{r.evidence}&rdquo;</span>}
                  </div>
                  <span className={`status ${r.status}`}>{STATUS[r.status]}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
      {soft.length > 0 && (
        <p className="soft">Also asked for, not counted: {soft.map((r) => r.text).join("; ")}. An interview is the place for these.</p>
      )}
      {data.failed.length > 0 && (
        <div className="error-box"><p className="error">Not ranked: {data.failed.map((f) => `${f.file} (${f.message})`).join("; ")}</p></div>
      )}
      <details className="how">
        <summary>Requirements used: {data.requirements.length - dropped.length - soft.length}</summary>
        <ul>{data.requirements.filter((r) => !r.issue && !r.soft).map((r, i) => <li key={i}>{r.text}{IMPORTANCE[r.importance] ? ` (${IMPORTANCE[r.importance]})` : ""}</li>)}</ul>
        {dropped.length > 0 && (
          <>
            <p className="muted small">Left out by the checks:</p>
            <ul>{dropped.map((r, i) => <li key={i}>{r.text}: {r.issue}</li>)}</ul>
          </>
        )}
      </details>
      <p className="disclaimer">{data.note}</p>
    </section>
  );
}

function RankCandidates() {
  const [job, setJob] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [blind, setBlind] = useState(true);
  const [reveal, setReveal] = useState(false);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<Ranking | null>(null);
  const input = useRef<HTMLInputElement>(null);

  async function run() {
    setWorking(true); setError(null); setData(null); setReveal(false);
    const body = new FormData();
    body.append("job_text", job);
    body.append("blind", String(blind));
    files.forEach((f) => body.append("files", f));
    try {
      const res = await fetch("/api/rank", { method: "POST", body });
      if (handleSignedOut(res)) return;
      const out = await res.json().catch(() => ({}));
      if (!res.ok) { setError(out?.detail?.message || "Something went wrong. Try again."); return; }
      setData(out);
      if (!out.blind) setReveal(true);
    } catch {
      setError("Couldn't reach the analysis service. Is the API running on port 8000?");
    } finally {
      setWorking(false);
    }
  }

  return (
    <>
      <p className="tag">Recruiter</p>
      <h1 className="serif">Rank candidates for a job</h1>
      <p className="lede">
        Paste a job description and upload up to 10 resumes. Each resume is checked against the same requirements
        the same way, with the evidence from its own text. Files are deleted once their text is read.
      </p>
      <div className="role-form">
        <label>
          Job description
          <textarea value={job} onChange={(e) => setJob(e.target.value)} rows={10} placeholder="Paste the full job description, including its requirements." />
        </label>
        <div>
          <div className="row">
            <button type="button" className="button ghost" onClick={() => input.current?.click()} disabled={working}>Choose resumes</button>
            <span className="small muted">{files.length ? `${files.length} selected: ${files.map((f) => f.name).join(", ")}` : "PDF or Word, up to 10"}</span>
          </div>
          <input ref={input} type="file" accept=".pdf,.docx" multiple hidden
                 onChange={(e) => { setFiles(Array.from(e.target.files ?? []).slice(0, 10)); e.target.value = ""; }} />
        </div>
        <label className="options">
          <input type="checkbox" checked={blind} onChange={(e) => setBlind(e.target.checked)} disabled={working} />
          Blind review: hide names, contact details, links, schools and graduation years
        </label>
        <div className="row">
          <button className="button" type="button" disabled={working || files.length === 0 || job.trim().length < 200} onClick={run}>
            {working ? "Ranking" : "Rank candidates"}
          </button>
          <span className="small muted">{job.trim().length < 200 ? "The job description needs at least 200 characters." : "Usually a few seconds."}</span>
        </div>
      </div>
      {error && <div className="error-box" role="alert"><p className="error">{error}</p></div>}
      {data && (
        <>
          {data.blind && (
            <p className="small" style={{ marginTop: 24 }}>
              {reveal
                ? <button type="button" className="link-button" onClick={() => setReveal(false)}>Hide names again</button>
                : <>Candidates are labeled by rank. <button type="button" className="link-button" onClick={() => setReveal(true)}>Reveal names</button> once you have a shortlist.</>}
            </p>
          )}
          <Results data={data} reveal={reveal} />
        </>
      )}
    </>
  );
}

export default function RecruitPage() {
  return <RoleGate href="/recruit"><RankCandidates /></RoleGate>;
}
