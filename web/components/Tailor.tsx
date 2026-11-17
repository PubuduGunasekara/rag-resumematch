"use client";

import { useState } from "react";
import type { Job } from "@/lib/types";
import { handleSignedOut } from "@/lib/session";

// Mirrors resumematch.api.app.tailor_to_json.
type Bullets = {
  kind: "bullets"; message: string | null;
  shown: { original: string; text: string; requirement: string }[];
  rejected: { original: string; text: string; issue: string }[];
};
type Letter = {
  kind: "letter"; message: string | null; left_out: number;
  paragraphs: { text: string; resume_quote: string | null; requirement: string | null }[][];
  rejected: { text: string; issue: string }[];
};

type PrepItem = { requirement: string; status: string; question: string; hint: string; generated: boolean; issue: string | null };
type Interview = { kind: "interview"; message: string | null; items: PrepItem[]; general: PrepItem[]; templates: number };

const GROUPS: { title: string; statuses: string[] }[] = [
  { title: "Questions about gaps, to rehearse first", statuses: ["not_found", "partly"] },
  { title: "Questions about what your resume shows", statuses: ["shown"] },
  { title: "General strengths", statuses: ["soft"] },
];

function InterviewView({ data }: { data: Interview }) {
  if (data.message) return <p className="small muted">{data.message}</p>;
  const failed = data.items.filter((i) => !i.generated && i.issue);
  const groups = GROUPS.map((g) => ({ ...g, items: data.items.filter((i) => g.statuses.includes(i.status)) }));
  if (data.general.length) groups.push({ title: "About the role", statuses: ["general"], items: data.general });
  return (
    <div className="tailor-out">
      {groups.filter((g) => g.items.length).map((g) => (
        <div key={g.title} className="prep-group">
          <p className="label">{g.title}</p>
          {g.items.map((i, n) => (
            <div className="rewrite" key={n}>
              <p className="question">{i.question}</p>
              {i.status !== "general" && <p className="small muted">For: {i.requirement}</p>}
              <p className="small">{i.hint}</p>
            </div>
          ))}
        </div>
      ))}
      <p className="small muted">
        Questions are what an interviewer is likely to ask; what to draw on comes from your resume and the checks above,
        not from the model.
        {data.templates > 0 && ` ${data.templates} ${data.templates === 1 ? "question is a plain template" : "questions are plain templates"} because the model's question failed a check or was missing.`}
      </p>
      {failed.length > 0 && (
        <details className="how">
          <summary>Why some model questions weren&apos;t used</summary>
          <ul>{failed.map((i, n) => <li key={n}>{i.requirement}: {i.issue}</li>)}</ul>
        </details>
      )}
    </div>
  );
}

function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
  const [done, setDone] = useState(false);
  return (
    <button type="button" className="link-button" onClick={() => {
      navigator.clipboard?.writeText(text).then(() => { setDone(true); setTimeout(() => setDone(false), 1500); }).catch(() => {});
    }}>{done ? "Copied" : label}</button>
  );
}

function BulletsView({ data }: { data: Bullets }) {
  if (data.message) return <p className="small muted">{data.message}</p>;
  return (
    <div className="tailor-out">
      {data.shown.length === 0 && <p className="small muted">Every suggestion added something your resume doesn&apos;t say, so none are shown. Try again for new suggestions.</p>}
      {data.shown.map((b, i) => (
        <div className="rewrite" key={i}>
          <p className="small muted">Your resume: {b.original}</p>
          <p>{b.text} <CopyButton text={b.text} /></p>
          {b.requirement && <p className="small muted">For: {b.requirement}</p>}
        </div>
      ))}
      {data.rejected.length > 0 && (
        <details className="how">
          <summary>{data.rejected.length} {data.rejected.length === 1 ? "suggestion was" : "suggestions were"} left out</summary>
          <ul>{data.rejected.map((r, i) => <li key={i}>{r.text} <span className="muted">({r.issue})</span></li>)}</ul>
        </details>
      )}
    </div>
  );
}

function LetterView({ data }: { data: Letter }) {
  if (data.message) return <p className="small muted">{data.message}</p>;
  const plain = data.paragraphs.map((p) => p.map((s) => s.text).join(" ")).join("\n\n");
  return (
    <div className="tailor-out">
      <div className="letter">
        {data.paragraphs.map((p, i) => (
          <p key={i}>
            {p.map((s, j) => (
              <span key={j}>
                {s.resume_quote
                  ? <span className="cited" title={`From your resume: "${s.resume_quote}"`}>{s.text}</span>
                  : s.text}{" "}
              </span>
            ))}
          </p>
        ))}
      </div>
      <p className="small"><CopyButton text={plain} label="Copy the letter" /></p>
      <p className="small muted">
        Underlined sentences come from your resume (hover to see the line). This is a draft to edit: tools, names
        and numbers were checked against your resume, ordinary phrasing was not.
        {data.left_out > 0 && ` ${data.left_out} repeated or extra general ${data.left_out === 1 ? "sentence was" : "sentences were"} left out.`}
      </p>
      {data.rejected.length > 0 && (
        <details className="how">
          <summary>{data.rejected.length} {data.rejected.length === 1 ? "sentence failed" : "sentences failed"} a check</summary>
          <ul>{data.rejected.map((r, i) => <li key={i}>{r.text} <span className="muted">({r.issue})</span></li>)}</ul>
        </details>
      )}
    </div>
  );
}

export default function Tailor({ job, resumeText }: { job: Job; resumeText: string }) {
  const [state, setState] = useState<{ kind: "idle" } | { kind: "working"; what: string } | { kind: "error"; message: string } | { kind: "done"; data: Bullets | Letter | Interview }>({ kind: "idle" });

  async function run(kind: "bullets" | "letter" | "interview") {
    const what = { bullets: "Rewriting your bullets for this posting", letter: "Drafting a cover letter", interview: "Preparing interview questions" }[kind];
    setState({ kind: "working", what });
    const evidence = job.requirements.filter((r) => r.met && r.resume_quote).map((r) => ({ requirement: r.text, resume_quote: r.resume_quote! }));
    const requirements = job.requirements.map((r) => ({ text: r.text, resume_quote: r.met ? r.resume_quote : null, credential: r.credential }));
    try {
      const res = kind === "interview"
        ? await fetch("/api/interview", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ job_id: job.job_id, resume_text: resumeText, requirements }),
          })
        : await fetch("/api/tailor", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ kind, job_id: job.job_id, resume_text: resumeText, evidence }),
          });
      if (handleSignedOut(res)) return;
      const body = await res.json().catch(() => ({}));
      if (!res.ok) { setState({ kind: "error", message: body?.detail?.message || "Something went wrong. Try again." }); return; }
      setState({ kind: "done", data: kind === "interview" ? { ...body, kind: "interview" } : body });
    } catch {
      setState({ kind: "error", message: "Couldn't reach the analysis service. Is the API running on port 8000?" });
    }
  }

  const working = state.kind === "working";
  return (
    <div className="tailor">
      <p className="tag">Tailor for this posting</p>
      <div className="row">
        <button type="button" className="button ghost" disabled={working} onClick={() => run("bullets")}>Suggest resume edits</button>
        <button type="button" className="button ghost" disabled={working} onClick={() => run("letter")}>Draft a cover letter</button>
        <button type="button" className="button ghost" disabled={working} onClick={() => run("interview")}>Prepare for the interview</button>
      </div>
      {working && <p className="small muted" role="status">{state.what}, usually 10 to 15 s.</p>}
      {state.kind === "error" && <p className="error" role="alert">{state.message}</p>}
      {state.kind === "done" && (state.data.kind === "bullets" ? <BulletsView data={state.data} />
        : state.data.kind === "letter" ? <LetterView data={state.data} /> : <InterviewView data={state.data} />)}
    </div>
  );
}
