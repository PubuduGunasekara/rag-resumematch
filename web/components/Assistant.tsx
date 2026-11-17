"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { NO_FILTERS } from "@/components/Filters";
import * as I from "@/components/Icons";
import { handleSignedOut, useSession } from "@/lib/session";
import { readError, useWorkspace, type TailorKind } from "@/lib/workspace";

// Mirrors resumematch.api.app.assistant (and chat_to_json for "qa").
type Action = { type: string; label: string; href?: string; job_id?: string; title?: string };
type QA = {
  answer: { text: string; sources: number[] }[]; message: string | null; by_code: boolean;
  sources: { id: number; label: string; text: string; url: string | null }[]; removed: { text: string; issue: string }[];
};
type RoleSummary = { title: string; analyzed: number; terms: { term: string; share: number; shown: boolean | null }[] };
type Reply = { kind: string; text: string[]; actions: Action[]; suggestions: string[]; answer?: QA; role?: RoleSummary };
type Message = { from: "you" | "bot"; text?: string; reply?: Reply; note?: string; error?: string };

const STORE = "rm_assistant_v1";   // the conversation, kept for this tab like the report

const TAILOR_NAMES: Record<TailorKind, string> = { bullets: "resume edits", letter: "cover letter", interview: "interview prep" };

function QAView({ qa }: { qa: QA }) {
  return (
    <>
      {qa.message && <p>{qa.message}</p>}
      {qa.answer.map((s, i) => <p key={i}>{s.text}{s.sources.length > 0 && <sup>{s.sources.join(",")}</sup>}</p>)}
      {qa.answer.length > 0 && (
        <p className="bot-meta">{qa.by_code ? "From your report's data." : "Written by the model, each sentence checked against its sources."}</p>
      )}
      {qa.sources.length > 0 && (
        <details className="bot-details">
          <summary>Sources ({qa.sources.length})</summary>
          <ol>{qa.sources.map((s) => <li key={s.id} value={s.id}><b>{s.label}:</b> {s.text}</li>)}</ol>
        </details>
      )}
      {qa.removed.length > 0 && (
        <details className="bot-details">
          <summary>Removed by the checks ({qa.removed.length})</summary>
          <ul>{qa.removed.map((r, i) => <li key={i}>{r.text} <i>({r.issue})</i></li>)}</ul>
        </details>
      )}
    </>
  );
}

function RoleView({ role }: { role: RoleSummary }) {
  return (
    <ul className="bot-terms">
      {role.terms.map((t) => (
        <li key={t.term} className={t.shown === true ? "yes" : t.shown === false ? "no" : ""}>
          <span>{t.term}</span>
          <span className="bot-bar"><i style={{ width: `${Math.round(t.share * 100)}%` }} /></span>
          <span className="bot-pct">{Math.round(t.share * 100)}%</span>
        </li>
      ))}
    </ul>
  );
}

const AUTO_OPEN_MS = 3000;          // after sign-in, once per tab
const TEASER_MS = 3000;             // on the login page: a teaser bubble, not the panel

/** Open the assistant from anywhere: window.dispatchEvent(new Event("rm:open-assistant")). */
export const OPEN_EVENT = "rm:open-assistant";

export default function Assistant({ guest = false }: { guest?: boolean }) {
  const router = useRouter();
  const { user } = useSession();
  const ws = useWorkspace();
  const [open, setOpen] = useState(false);
  const [unread, setUnread] = useState(false);
  const [teaser, setTeaser] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const inputBox = useRef<HTMLInputElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const hadReport = useRef<boolean | null>(null);
  const lastQA = useRef<{ question: string; answer: string } | null>(null);
  const seen = useRef(0);

  useEffect(() => { scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" }); }, [messages, busy, open]);

  // A new reply while the panel is closed shows a dot on the launcher.
  useEffect(() => {
    if (open) { setUnread(false); seen.current = messages.length; return; }
    if (messages.length > seen.current && messages[messages.length - 1]?.from === "bot") setUnread(true);
  }, [messages, open]);

  // Once per tab: the panel opens by itself a few seconds after sign-in. On the
  // login page only a teaser bubble appears, because the open panel covered the
  // sign-in form at common screen widths.
  useEffect(() => {
    const key = guest ? "rm_chat_auto_guest" : "rm_chat_auto";
    try { if (sessionStorage.getItem(key)) return; } catch { /* ignore */ }
    const t = setTimeout(() => {
      try { sessionStorage.setItem(key, "1"); } catch { /* ignore */ }
      if (guest) setTeaser(true); else setOpen(true);
    }, guest ? TEASER_MS : AUTO_OPEN_MS);
    return () => clearTimeout(t);
  }, [guest]);

  useEffect(() => {
    const show = () => setOpen(true);
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener(OPEN_EVENT, show);
    window.addEventListener("keydown", esc);
    return () => { window.removeEventListener(OPEN_EVENT, show); window.removeEventListener("keydown", esc); };
  }, []);
  useEffect(() => { if (open) { setTeaser(false); setTimeout(() => inputBox.current?.focus(), 50); } }, [open]);

  // Restore a signed-in conversation after a reload; save it as it grows (last 60 messages).
  const restored = useRef(false);
  useEffect(() => {
    if (guest) { restored.current = true; return; }
    try {
      const saved = JSON.parse(sessionStorage.getItem(STORE) || "null");
      if (saved?.user === user?.email && Array.isArray(saved.messages) && saved.messages.length) {
        setMessages(saved.messages);
        seen.current = saved.messages.length;
        hadReport.current = saved.hadReport;
      }
    } catch { /* start fresh */ }
    restored.current = true;
  }, [user?.email, guest]);
  useEffect(() => {
    if (guest || !restored.current || !user) return;
    try { sessionStorage.setItem(STORE, JSON.stringify({ user: user.email, hadReport: hadReport.current, messages: messages.slice(-60) })); } catch { /* ignore */ }
  }, [messages, user, guest]);

  async function send(message: string, show = true) {
    const text = message.trim();
    if (!text || busy) return;
    if (show) setMessages((m) => [...m, { from: "you", text }]);
    setBusy(true);
    try {
      let res: Response;
      if (guest) {
        res = await fetch("/api/assistant/guest", {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: text }),
        });
      } else {
        const report = ws.report;
        const jobs = report ? report.jobs.map((j) => ({
          job_id: j.job_id, title: j.title, company: j.company, closest: j.job_id === report.best_job_id,
          requirements: j.requirements.map((r) => ({ text: r.text, status: r.status, resume_quote: r.met ? r.resume_quote : null, soft: r.soft })),
        })) : [];
        res = await fetch("/api/assistant", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, resume_text: report?.resume_text ?? "", jobs, history: lastQA.current ? [lastQA.current] : [] }),
        });
        if (handleSignedOut(res)) return;
      }
      if (!res.ok) { const { message } = await readError(res); setMessages((m) => [...m, { from: "bot", error: message }]); return; }
      const reply: Reply = await res.json();
      if (reply.kind === "qa" && reply.answer?.answer.length) {
        lastQA.current = { question: text, answer: reply.answer.answer.map((s) => s.text).join(" ") };
      }
      setMessages((m) => [...m, { from: "bot", reply }]);
      const auto = reply.actions.find((a) => a.type === "bullets" || a.type === "letter" || a.type === "interview");
      if (auto?.job_id) await runTailor(auto.job_id, auto.type as TailorKind, auto.label);
    } catch {
      setMessages((m) => [...m, { from: "bot", error: "Couldn't reach the analysis service. Is the API running on port 8000?" }]);
    } finally {
      setBusy(false);
    }
  }

  async function runTailor(jobId: string, kind: TailorKind, label: string) {
    const result = await ws.runTailor(jobId, kind);
    if (result.ok) {
      setMessages((m) => [...m, { from: "bot", note: `Your ${TAILOR_NAMES[kind]} for ${label} is ready.` }]);
      router.push(`/tailor?job=${encodeURIComponent(jobId)}&tab=${kind}`);
    } else if (result.message) {
      setMessages((m) => [...m, { from: "bot", error: result.message }]);
    }
  }

  async function upload(file: File) {
    setMessages((m) => [...m, { from: "you", text: `Uploaded ${file.name}` }]);
    setBusy(true);
    const ok = await ws.analyze(file, { correction: true, filters: NO_FILTERS });
    setBusy(false);
    if (!ok) setMessages((m) => [...m, { from: "bot", error: "That file couldn't be checked. The Start page shows why." }]);
    else router.push("/");
  }

  // Greeting: the server's text for this visitor, so it only offers what exists.
  // Signed in, it greets again when a report arrives.
  useEffect(() => {
    if (!restored.current) return;
    if (guest) { if (!messages.length) send("hello", false); return; }
    if (!user) return;
    const has = !!ws.report;
    if (hadReport.current === has) return;
    const first = hadReport.current === null;
    hadReport.current = has;
    if (first || has) send("help", false);
  }, [user, ws.report, guest]);   // eslint-disable-line react-hooks/exhaustive-deps

  function act(a: Action) {
    if (a.type === "upload") fileInput.current?.click();
    else if (a.type === "open" && a.href) router.push(a.href);
    else if (a.job_id && (a.type === "bullets" || a.type === "letter" || a.type === "interview")) {
      if (ws.tailored[a.job_id]?.[a.type as TailorKind]) router.push(`/tailor?job=${encodeURIComponent(a.job_id)}&tab=${a.type}`);
      else runTailor(a.job_id, a.type as TailorKind, a.label);
    }
  }

  const lastBot = [...messages].reverse().find((m) => m.reply)?.reply;
  const working = busy || !!ws.tailoring || !!ws.analyzing;
  const canUpload = !guest && !!user && user.role !== "recruiter";

  return (
    <>
      {teaser && !open && (
        <button type="button" className="chat-teaser" onClick={() => setOpen(true)}>
          Hi! Questions about ResumeMatch? <I.Close size={13} className="teaser-x" />
        </button>
      )}
      <button type="button" className={`chat-launcher${open ? " open" : ""}`} onClick={() => setOpen(!open)}
              aria-label={open ? "Close the assistant" : "Open the assistant"} aria-expanded={open} aria-controls="assistant">
        {open ? <I.Close size={22} /> : <I.Chat size={24} />}
        {unread && !open && <span className="chat-unread" aria-label="New message" />}
      </button>

      <section id="assistant" className={`chat-panel${open ? " open" : ""}`} aria-label="Assistant" aria-hidden={!open} role="dialog">
        <header className="assistant-head">
          <span className="assistant-dot" aria-hidden="true"><I.Spark size={15} /></span>
          <div>
            <b>ResumeMatch assistant</b>
            <span>{guest ? "Sign in to use it with your resume" : "Answers from your resume and real postings"}</span>
          </div>
          <button type="button" className="icon-button" onClick={() => setOpen(false)} aria-label="Minimize the assistant"><I.Close size={17} /></button>
        </header>

        <div className="assistant-log" ref={scroller} aria-live="polite">
          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.from}`}>
              {m.text && <p>{m.text}</p>}
              {m.note && <p className="bot-note"><I.Check size={14} /> {m.note}</p>}
              {m.error && <p className="bot-error">{m.error}</p>}
              {m.reply && (
                <>
                  {m.reply.text.map((t, j) => <p key={j}>{t}</p>)}
                  {m.reply.role && <RoleView role={m.reply.role} />}
                  {m.reply.answer && <QAView qa={m.reply.answer} />}
                  {m.reply.actions.filter((a) => a.type === "upload" || a.type === "open").length > 0 && (
                    <div className="bot-actions">
                      {m.reply.actions.filter((a) => a.type === "upload" || a.type === "open").map((a, j) => (
                        <button type="button" key={j} className={a.type === "upload" ? "chip primary" : "chip"} onClick={() => act(a)} disabled={working && a.type === "upload"}>
                          {a.type === "upload" ? <I.Upload size={14} /> : <I.Arrow size={14} />} {a.label}
                        </button>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          ))}
          {working && (
            <div className="msg bot typing" role="status">
              <span /><span /><span />
              <em>{ws.analyzing ? "Checking your resume against the postings, usually about 20 s" : ws.tailoring ? `Writing your ${TAILOR_NAMES[ws.tailoring.kind]}` : "Thinking"}</em>
            </div>
          )}
        </div>

        {lastBot && lastBot.suggestions.length > 0 && !working && (
          <div className="assistant-suggest">
            {lastBot.suggestions.map((s) => <button type="button" key={s} className="chip" onClick={() => send(s)}>{s}</button>)}
          </div>
        )}

        <form className="assistant-input" onSubmit={(e) => { e.preventDefault(); send(input); setInput(""); }}>
          {canUpload && (
            <button type="button" className="icon-button" onClick={() => fileInput.current?.click()} aria-label="Upload a resume" title="Upload a resume" disabled={working}>
              <I.Upload size={17} />
            </button>
          )}
          <input ref={inputBox} value={input} onChange={(e) => setInput(e.target.value)} maxLength={400}
                 placeholder={guest ? "Ask about ResumeMatch" : "Ask anything about your job search"} aria-label="Message the assistant" />
          <button type="submit" className="send" disabled={working || input.trim().length < 1} aria-label="Send"><I.Send size={16} /></button>
          <input ref={fileInput} type="file" accept=".pdf,.docx" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) upload(f); e.target.value = ""; }} />
        </form>
      </section>
    </>
  );
}
