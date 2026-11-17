"use client";

import { useRef, useState } from "react";
import type { Report } from "@/lib/types";
import { handleSignedOut } from "@/lib/session";

// Mirrors resumematch.api.app.comparison_to_json.
type Change = { requirement: string; before: string; after: string; better: boolean; evidence: string | null };
type Result = {
  summary: string; note: string; total: number; shown_before: number; shown_after: number;
  jobs: { job_id: string; title: string; company: string | null; total: number; shown_before: number; shown_after: number; changes: Change[] }[];
  added_lines: string[]; removed_lines: string[];
};

export default function Compare({ report }: { report: Report }) {
  const input = useRef<HTMLInputElement>(null);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);

  async function run(file: File) {
    setWorking(true); setError(null); setResult(null);
    const jobs = report.jobs.map((j) => ({ job_id: j.job_id, title: j.title, company: j.company, requirements: j.requirements.map((r) => r.text) }));
    const body = new FormData();
    body.append("file", file);
    body.append("old_resume_text", report.resume_text);
    body.append("jobs", JSON.stringify(jobs));
    try {
      const res = await fetch("/api/compare", { method: "POST", body });
      if (handleSignedOut(res)) return;
      const data = await res.json().catch(() => ({}));
      if (!res.ok) { setError(data?.detail?.message || "Something went wrong. Try again."); return; }
      setResult(data);
    } catch {
      setError("Couldn't reach the analysis service. Is the API running on port 8000?");
    } finally {
      setWorking(false);
    }
  }

  return (
    <div className="section">
      <p className="tag">Compare a revised version</p>
      <p className="small muted" style={{ marginTop: 0 }}>
        Edited your resume? Upload the new version to see which of these postings&apos; requirements it now shows.
        The same postings are used, so the two versions can be compared.
      </p>
      <div className="row">
        <button type="button" className="button ghost" disabled={working} onClick={() => input.current?.click()}>
          {working ? "Comparing" : "Choose the revised file"}
        </button>
        <input ref={input} type="file" accept=".pdf,.docx" hidden
               onChange={(e) => { const f = e.target.files?.[0]; if (f) run(f); e.target.value = ""; }} />
      </div>
      {error && <p className="error" role="alert" style={{ marginTop: 10 }}>{error}</p>}
      {result && (
        <div className="compare-out">
          <p className="compare-summary">{result.summary}</p>
          {result.jobs.filter((j) => j.total > 0).map((j) => (
            <div className="compare-job" key={j.job_id}>
              <p className="compare-head">
                <span>{j.title}{j.company ? `, ${j.company}` : ""}</span>
                <span className="muted">{j.shown_before} to {j.shown_after} of {j.total} shown</span>
              </p>
              {j.changes.length === 0 ? <p className="small muted">No change.</p> : (
                <ul>
                  {j.changes.map((c, i) => (
                    <li key={i} className={c.better ? "better" : "worse"}>
                      {c.requirement}: {c.before} to <b style={{ fontWeight: 500 }}>{c.after}</b>
                      {c.evidence && <span className="quote">{c.better ? "Now on your resume" : "Was on your resume"}: &ldquo;{c.evidence}&rdquo;</span>}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}
          {(result.added_lines.length > 0 || result.removed_lines.length > 0) && (
            <details className="how">
              <summary>Lines added and removed</summary>
              <ul>
                {result.added_lines.map((l, i) => <li key={`a${i}`}>Added: {l}</li>)}
                {result.removed_lines.map((l, i) => <li key={`r${i}`}>Removed: {l}</li>)}
              </ul>
            </details>
          )}
          <p className="small muted">{result.note}</p>
        </div>
      )}
    </div>
  );
}
