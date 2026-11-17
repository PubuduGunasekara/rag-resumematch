"use client";

import { useEffect, useRef, useState } from "react";
import type { RoleReport } from "@/lib/types";
import { handleSignedOut } from "@/lib/session";

type Suggestion = { title: string; postings: number };

async function readError(res: Response): Promise<string> {
  try {
    const detail = (await res.json()).detail;
    return (detail && typeof detail === "object" && detail.message) || "Something went wrong. Try again in a moment.";
  } catch {
    return "Something went wrong. Try again in a moment.";
  }
}

function Facts({ report }: { report: RoleReport }) {
  const f = report.facts;
  const items: string[] = [];
  if (f.remote_share !== undefined) items.push(`${Math.round(f.remote_share * 100)}% allow remote work`);
  if (f.salary_listed_share !== undefined) items.push(`${Math.round(f.salary_listed_share * 100)}% list a salary`);
  if (f.top_locations?.length) items.push(`Most often in ${f.top_locations.slice(0, 3).map((l) => l.name).join("; ")}`);
  if (!items.length) return null;
  return <div className="facts">{items.map((i) => <span key={i}>{i}</span>)}</div>;
}

export default function RolePage() {
  const [role, setRole] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [report, setReport] = useState<RoleReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [working, setWorking] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  // Title suggestions from the index, after a short pause in typing.
  useEffect(() => {
    if (role.trim().length < 2) { setSuggestions([]); return; }
    const timer = setTimeout(() => {
      fetch(`/api/roles?q=${encodeURIComponent(role.trim())}`)
        .then((r) => (r.ok ? r.json() : { suggestions: [] }))
        .then((b) => setSuggestions(b.suggestions ?? []))
        .catch(() => setSuggestions([]));
    }, 250);
    return () => clearTimeout(timer);
  }, [role]);

  async function run(title: string) {
    if (title.trim().length < 2) { setError("Enter a job title, for example Data Engineer."); return; }
    setWorking(true); setError(null);
    const body = new FormData();
    body.append("role", title.trim());
    if (file) body.append("file", file);
    try {
      const res = await fetch("/api/role-report", { method: "POST", body });
      if (handleSignedOut(res)) return;
      if (!res.ok) { setError(await readError(res)); setReport(null); return; }
      setReport(await res.json());
      setSuggestions([]);
    } catch {
      setError("Couldn't reach the analysis service. Is the API running on port 8000?");
    } finally {
      setWorking(false);
    }
  }

  return (
    <>
      <p className="tag">Role report</p>
      <h1 className="serif">See what a job title asks for, across every posting</h1>
      <p className="lede">
        Enter a job title. ResumeMatch reads every indexed posting with that title and lists the terms their
        requirements use most, compared with postings in general. Add your resume to see which of them it uses.
      </p>

      <form className="role-form" onSubmit={(e) => { e.preventDefault(); run(role); }}>
        <label>
          Job title
          <input type="text" value={role} onChange={(e) => setRole(e.target.value)} placeholder="Data Engineer"
                 autoComplete="off" maxLength={80} />
        </label>
        {suggestions.length > 0 && (
          <div className="suggest" aria-label="Titles in the index">
            {suggestions.map((s) => (
              <button type="button" key={s.title} onClick={() => { setRole(s.title); run(s.title); }}>
                {s.title} <span className="muted">({s.postings.toLocaleString()})</span>
              </button>
            ))}
          </div>
        )}
        <label>
          Your resume (optional, PDF or Word)
          <div className="row">
            <input ref={fileInput} type="file" accept=".pdf,.docx" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            {file && <button type="button" className="button ghost" onClick={() => { setFile(null); if (fileInput.current) fileInput.current.value = ""; }}>Remove</button>}
          </div>
        </label>
        <div className="row">
          <button className="button" type="submit" disabled={working}>{working ? "Reading postings" : "Show the report"}</button>
          <span className="small muted">No language model is used here; it takes a few seconds.</span>
        </div>
      </form>

      {error && <div className="error-box" role="alert"><p className="error">{error}</p></div>}

      {report && (
        <section className="report" aria-label={`Role report for ${report.role}`}>
          <h1 className="serif">{report.headline}</h1>
          <p className="lede">{report.summary}</p>
          <Facts report={report} />
          {report.matched > report.analyzed && report.analyzed > 0 && (
            <p className="small muted">{report.matched.toLocaleString()} postings have this title; a fixed sample of {report.analyzed.toLocaleString()} was read.</p>
          )}

          {report.terms.length > 0 && (
            <div className="section">
              <p className="tag">Terms in these postings&apos; requirements</p>
              {report.terms.map((t) => (
                <div className="term" key={t.term}>
                  <div className="term-row">
                    <span>{t.term}</span>
                    <span>
                      <div className="meter"><div style={{ width: `${Math.round(t.share * 100)}%` }} /></div>
                      <span className="pct">{Math.round(t.share * 100)}% of postings</span>
                    </span>
                    <span className="status-cell">
                      {t.shown === true && <span className="status shown">On your resume</span>}
                      {t.shown === false && <span className="status not_found">Not on your resume</span>}
                    </span>
                  </div>
                  {t.examples.length > 0 && (
                    <details>
                      <summary>Example postings</summary>
                      <ul>
                        {t.examples.map((e) => (
                          <li key={e.job_id}>
                            {e.url ? <a href={e.url} target="_blank" rel="noopener noreferrer">{e.title}</a> : e.title}
                            {e.company ? `, ${e.company}` : ""}
                          </li>
                        ))}
                      </ul>
                    </details>
                  )}
                </div>
              ))}
            </div>
          )}

          <p className="disclaimer">
            Terms are counted from the wording of requirement sentences, so a skill phrased differently on your resume
            shows as not on it. Postings are from 2023 and 2024.
          </p>
        </section>
      )}
    </>
  );
}
