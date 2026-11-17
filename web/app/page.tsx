"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import Filters, { FilterNote, NO_FILTERS, type FilterValues } from "@/components/Filters";
import * as I from "@/components/Icons";
import { Facts, FitRing } from "@/components/ReportParts";
import RoleGate from "@/components/RoleGate";
import { useWorkspace } from "@/lib/workspace";

type Problem = { code: string; message: string; blocking: boolean };

const EXPECT = [
  { icon: I.Briefcase, title: "Your closest postings", text: "The postings closest to your whole resume, searched section by section, with each requirement marked shown, partly shown or not found." },
  { icon: I.FileText, title: "Evidence in your resume", text: "Click any requirement to see the exact line of your resume that shows it. Nothing is counted without a line you wrote." },
  { icon: I.Pen, title: "Edits and a cover letter", text: "Suggested rewrites of your bullets and a cover letter draft for any posting. Anything that adds a fact you didn't state is removed." },
  { icon: I.Chat, title: "Interview prep and answers", text: "Likely interview questions with what to draw on, and an assistant that answers from your resume and the postings, with sources." },
];

const STEPS = ["Reading your resume", "Searching the postings, section by section", "Checking each requirement against your resume", "Building your report"];

function Analyzing({ name, started }: { name: string; started: number }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setElapsed(Math.round((Date.now() - started) / 1000)), 500);
    return () => clearInterval(t);
  }, [started]);
  // The pipeline doesn't report progress; steps advance on a timer that matches
  // a typical run (about 20 s), and the last one waits for the real answer.
  const step = Math.min(STEPS.length - 1, Math.floor(elapsed / 5));
  return (
    <div className="card analyzing" role="status" aria-live="polite">
      <div className="analyzing-head">
        <span className="spinner" aria-hidden="true" />
        <div><b>Checking {name}</b><span className="muted"> {elapsed} s, usually about 20</span></div>
      </div>
      <ol className="steps">
        {STEPS.map((s, i) => <li key={s} className={i < step ? "done" : i === step ? "now" : ""}>{i < step ? <I.Check size={14} /> : <span className="dot" />}{s}</li>)}
      </ol>
    </div>
  );
}

function Onboarding() {
  const ws = useWorkspace();
  const [over, setOver] = useState(false);
  const [correction, setCorrection] = useState(true);
  const [filters, setFilters] = useState<FilterValues>(NO_FILTERS);
  const [problems, setProblems] = useState<Problem[]>([]);
  const [postings, setPostings] = useState<number | null>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch("/api/health").then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((h) => { setProblems(h.problems ?? []); setPostings(h.postings ?? null); })
      .catch(() => setProblems([{ code: "api_down", blocking: true, message: "Can't reach the analysis service. Start it with: uvicorn resumematch.api.app:app --port 8000" }]));
  }, []);
  const blocked = problems.some((p) => p.blocking);
  const disabled = blocked || !!ws.analyzing;
  const go = (file: File, allowUnusual = false) => ws.analyze(file, { correction, filters, allowUnusual });

  return (
    <div className="start">
      <section className="hero">
        <p className="eyebrow">Resume check</p>
        <h1>See which real jobs your resume already fits, and what to change.</h1>
        <p className="hero-sub">
          Upload your resume. ResumeMatch searches {postings ? `${postings.toLocaleString()} LinkedIn postings` : "the indexed LinkedIn postings"},
          checks every requirement of the closest ones against what your resume actually says, and helps you tailor it.
        </p>
      </section>

      {problems.map((p) => <p key={p.code} className={p.blocking ? "banner error" : "banner"}>{p.message}</p>)}

      {ws.analyzing ? <Analyzing name={ws.analyzing.name} started={ws.analyzing.started} /> : (
        <div className={`dropzone${over ? " over" : ""}${disabled ? " disabled" : ""}`}
             onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
             onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f && !disabled) go(f); }}>
          <span className="drop-icon"><I.Upload size={26} /></span>
          <div className="drop-text">
            <b>Drop your resume here</b>
            <span>PDF or Word, up to 5 MB. Deleted as soon as its text is read.</span>
          </div>
          <button type="button" className="btn primary" disabled={disabled} onClick={() => input.current?.click()}>Choose a file</button>
          <input ref={input} type="file" accept=".pdf,.docx" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) go(f); e.target.value = ""; }} />
        </div>
      )}

      {ws.failure && (
        <div className="banner error" role="alert">
          {ws.failure.message}
          {ws.failure.soft && ws.failure.file && <button type="button" className="btn ghost small" onClick={() => go(ws.failure!.file!, true)}>Check it anyway</button>}
        </div>
      )}

      <div className="start-options">
        <label className="switch">
          <input type="checkbox" checked={correction} onChange={(e) => setCorrection(e.target.checked)} disabled={disabled} />
          <span>Fix typos before searching <em>(your resume is still checked as written)</em></span>
        </label>
        <Filters value={filters} onChange={setFilters} disabled={disabled} />
      </div>

      <section className="expect">
        <h2 className="section-title">What you get</h2>
        <div className="expect-grid">
          {EXPECT.map(({ icon: Icon, title, text }) => (
            <div className="card expect-card" key={title}>
              <span className="expect-icon"><Icon size={20} /></span>
              <h3>{title}</h3>
              <p>{text}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="honest">
        <div><b>Checked, not guessed.</b> A local language model reads the postings; code checks every claim against your resume and the posting before you see it.</div>
        <div><b>Private.</b> The model runs on this server. Your file is deleted after reading, and your report lives only in this browser tab.</div>
        <div><b>Honest about limits.</b> Checks compare wording, so a skill written differently can show as not found. Add it if it&apos;s true.</div>
      </section>
    </div>
  );
}

function Overview() {
  const { report, fileName, clear } = useWorkspace();
  if (!report) return null;
  const best = report.jobs.find((j) => j.job_id === report.best_job_id) ?? report.jobs[0];
  const others = report.jobs.filter((j) => j.job_id !== best?.job_id);
  return (
    <div className="overview">
      <section className="hero compact">
        <p className="eyebrow">Your report {fileName ? `for ${fileName}` : ""}</p>
        <h1>{report.headline}</h1>
        <p className="hero-sub">{report.summary}{report.total_postings ? ` Searched ${report.total_postings.toLocaleString()} LinkedIn postings.` : ""}</p>
        <FilterNote filters={report.filters} />
      </section>

      {best && (
        <section className="card match-hero">
          <FitRing job={best} size={96} />
          <div className="match-hero-body">
            <p className="eyebrow">{best.tier === "no" ? "Closest posting" : "Closest match"}</p>
            <h2>{best.title}</h2>
            <p className="muted">{best.company}</p>
            <Facts details={best.details} />
            <p className={`reason${best.tier === "no" ? " warn" : ""}`}>{best.reason}</p>
            <div className="btn-row">
              <Link className="btn primary" href={`/matches?job=${best.job_id}`}>See the checklist</Link>
              <Link className="btn ghost" href={`/tailor?job=${best.job_id}&tab=bullets`}>Tailor my resume</Link>
              <Link className="btn ghost" href={`/tailor?job=${best.job_id}&tab=letter`}>Draft a cover letter</Link>
            </div>
          </div>
        </section>
      )}

      <div className="overview-grid">
        <section className="card">
          <h3 className="card-title">Other postings</h3>
          {others.length === 0 && <p className="muted small">No other postings in this report.</p>}
          {others.map((j) => (
            <Link key={j.job_id} href={`/matches?job=${j.job_id}`} className="mini-match">
              <FitRing job={j} size={44} />
              <span className="mini-text"><b>{j.title}</b><span>{j.company}</span></span>
              <span className={`tier ${j.tier}`}>{j.tier === "no" ? "Not yet" : j.tier === "best" ? "Best" : "Worth a look"}</span>
            </Link>
          ))}
        </section>
        <section className="card">
          <h3 className="card-title">What to work on</h3>
          {report.learn_first.length === 0 ? <p className="muted small">Nothing came up in more than one posting.</p> : report.learn_first.map((item) => (
            <div className="learn-row" key={item.text}>
              <span>{item.text}{item.credential && <em className="tag-credential">credential</em>}</span>
              <span className="bar"><i style={{ width: `${(100 * item.jobs) / report.jobs.length}%` }} /></span>
              <span className="muted small">{item.jobs} of {report.jobs.length}</span>
            </div>
          ))}
          <div className="next-steps">
            <Link href="/resume" className="next"><I.FileText size={16} /> See the evidence in your resume <I.Arrow size={14} /></Link>
            <Link href="/compare" className="next"><I.Layers size={16} /> Edited it? Compare a revised version <I.Arrow size={14} /></Link>
          </div>
        </section>
      </div>

      <button type="button" className="btn ghost small" onClick={clear}>Start over with another resume</button>
      <p className="fineprint">Checks compare wording, so a requirement marked not found may still be something you have. Add it to your resume if it&apos;s true.</p>
    </div>
  );
}

function StartPage() {
  const { report } = useWorkspace();
  return report ? <Overview /> : <Onboarding />;
}

export default function Home() {
  return <RoleGate href="/"><StartPage /></RoleGate>;
}
