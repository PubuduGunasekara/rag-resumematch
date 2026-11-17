"use client";

import { useState } from "react";
import type { Report } from "@/lib/types";
import { handleSignedOut } from "@/lib/session";

// Mirrors resumematch.api.app.chat_to_json.
type Source = { id: number; kind: string; label: string; text: string; url: string | null };
type Reply = {
  answer: { text: string; sources: number[] }[]; message: string | null; no_answer: boolean; by_code: boolean;
  sources: Source[]; removed: { text: string; issue: string }[];
};
type Turn = { question: string; reply: Reply | null; error: string | null };

const SUGGESTIONS = [
  "Why is my closest match a good fit?",
  "Which of these jobs list a salary?",
  "What experience do I have that these jobs ask for?",
];

function ReplyView({ reply }: { reply: Reply }) {
  return (
    <div className="chat-reply">
      {reply.message && <p className="muted">{reply.message}</p>}
      {reply.answer.map((s, i) => (
        <p key={i}>{s.text}{s.sources.length > 0 && <sup className="muted"> {s.sources.join(", ")}</sup>}</p>
      ))}
      {reply.answer.length > 0 && !reply.no_answer && (
        <p className="small muted">
          {reply.by_code ? "Answered directly from the report's data." : "Written by the model; each sentence was checked against its sources."}
        </p>
      )}
      {reply.sources.length > 0 && (
        <details className="how">
          <summary>Sources: {reply.sources.length}</summary>
          <ul>
            {reply.sources.map((s) => (
              <li key={s.id}>
                <b style={{ fontWeight: 500 }}>{s.id}. {s.url ? <a href={s.url} target="_blank" rel="noopener noreferrer">{s.label}</a> : s.label}:</b> {s.text}
              </li>
            ))}
          </ul>
        </details>
      )}
      {reply.removed.length > 0 && (
        <details className="how">
          <summary>{reply.removed.length} {reply.removed.length === 1 ? "sentence was" : "sentences were"} removed by the checks</summary>
          <ul>{reply.removed.map((r, i) => <li key={i}>{r.text} <span className="muted">({r.issue})</span></li>)}</ul>
        </details>
      )}
    </div>
  );
}

export default function Chat({ report }: { report: Report }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [working, setWorking] = useState(false);

  async function ask(text: string) {
    const q = text.trim();
    if (q.length < 3 || working) return;
    setWorking(true); setQuestion("");
    const last = [...turns].reverse().find((t) => t.reply && t.reply.answer.length > 0);
    const history = last ? [{ question: last.question, answer: last.reply!.answer.map((s) => s.text).join(" ") }] : [];
    const jobs = report.jobs.map((j) => ({
      job_id: j.job_id, closest: j.job_id === report.best_job_id,
      requirements: j.requirements.map((r) => ({ text: r.text, status: r.status, resume_quote: r.met ? r.resume_quote : null, soft: r.soft })),
    }));
    let turn: Turn = { question: q, reply: null, error: null };
    try {
      const res = await fetch("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, resume_text: report.resume_text, jobs, history }),
      });
      if (handleSignedOut(res)) return;
      const body = await res.json().catch(() => ({}));
      turn = res.ok ? { ...turn, reply: body } : { ...turn, error: body?.detail?.message || "Something went wrong. Try again." };
    } catch {
      turn = { ...turn, error: "Couldn't reach the analysis service. Is the API running on port 8000?" };
    } finally {
      setWorking(false);
    }
    setTurns((t) => [...t, turn]);
  }

  return (
    <div className="section">
      <p className="tag">Ask about this report</p>
      <p className="small muted" style={{ marginTop: 0 }}>
        Answers come only from your resume and these postings, with the sources listed. The conversation stays in
        this page and is gone when you leave it.
      </p>
      {turns.map((t, i) => (
        <div className="chat-turn" key={i}>
          <p className="chat-q">{t.question}</p>
          {t.error ? <p className="error">{t.error}</p> : t.reply && <ReplyView reply={t.reply} />}
        </div>
      ))}
      {working && <p className="small muted" role="status">Looking through your resume and the postings.</p>}
      {turns.length === 0 && (
        <div className="suggest">
          {SUGGESTIONS.map((s) => <button type="button" key={s} disabled={working} onClick={() => ask(s)}>{s}</button>)}
        </div>
      )}
      <form className="chat-form" onSubmit={(e) => { e.preventDefault(); ask(question); }}>
        <input type="text" value={question} onChange={(e) => setQuestion(e.target.value)} maxLength={400}
               placeholder="For example: does the second posting need a license?" aria-label="Your question" disabled={working} />
        <button className="button" type="submit" disabled={working || question.trim().length < 3}>Ask</button>
      </form>
    </div>
  );
}
