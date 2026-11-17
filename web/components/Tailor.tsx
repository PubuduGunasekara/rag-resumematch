"use client";

import { useState } from "react";

// Mirrors resumematch.api.app.tailor_to_json.
export type Bullets = {
  kind: "bullets"; message: string | null;
  shown: { original: string; text: string; requirement: string }[];
  rejected: { original: string; text: string; issue: string }[];
};
export type Letter = {
  kind: "letter"; message: string | null; left_out: number;
  paragraphs: { text: string; resume_quote: string | null; requirement: string | null }[][];
  rejected: { text: string; issue: string }[];
};

type PrepItem = { requirement: string; status: string; question: string; hint: string; generated: boolean; issue: string | null };
export type Interview = { kind: "interview"; message: string | null; items: PrepItem[]; general: PrepItem[]; templates: number };

const GROUPS: { title: string; statuses: string[] }[] = [
  { title: "Questions about gaps, to rehearse first", statuses: ["not_found", "partly"] },
  { title: "Questions about what your resume shows", statuses: ["shown"] },
  { title: "General strengths", statuses: ["soft"] },
];

export function InterviewView({ data }: { data: Interview }) {
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

export function BulletsView({ data }: { data: Bullets }) {
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

export function LetterView({ data }: { data: Letter }) {
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
