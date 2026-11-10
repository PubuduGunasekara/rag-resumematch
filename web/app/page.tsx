"use client";

import { useEffect, useRef, useState } from "react";
import ReportView from "@/components/Report";
import type { Report } from "@/lib/types";

type State =
  | { kind: "idle" }
  | { kind: "working"; name: string; started: number }
  | { kind: "done"; report: Report; name: string }
  | { kind: "error"; message: string; file?: File; soft?: boolean };

// The API answers errors as {detail: {code, message, soft}}; anything else
// (a proxy error page, a crash) gets a plain fallback, never raw output.
async function readError(res: Response): Promise<{ message: string; soft: boolean }> {
  const fallback = "Something went wrong on the server. Try again in a moment.";
  try {
    const detail = (await res.json()).detail;
    if (detail && typeof detail === "object") return { message: detail.message || fallback, soft: !!detail.soft };
    return { message: typeof detail === "string" ? detail : fallback, soft: false };
  } catch {
    return { message: fallback, soft: false };
  }
}

const ACCEPT = [".pdf", ".docx"];

type Problem = { code: string; message: string; blocking: boolean };

export default function Home() {
  const [state, setState] = useState<State>({ kind: "idle" });
  const [correction, setCorrection] = useState(true);
  const [over, setOver] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const [problems, setProblems] = useState<Problem[]>([]);

  // Startup check: say what's missing (index, Ollama, model) before anyone uploads.
  useEffect(() => {
    fetch("/api/health")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((h) => setProblems(h.problems ?? []))
      .catch(() => setProblems([{ code: "api_down", blocking: true,
        message: "Can't reach the analysis service. Start it with: uvicorn resumematch.api.app:app --port 8000" }]));
  }, []);
  const blocked = problems.some((p) => p.blocking);

  useEffect(() => {
    if (state.kind !== "working") return;
    const timer = setInterval(() => setElapsed(Math.round((Date.now() - state.started) / 1000)), 1000);
    return () => clearInterval(timer);
  }, [state]);

  async function analyze(file: File, allowUnusual = false) {
    if (!ACCEPT.some((ext) => file.name.toLowerCase().endsWith(ext))) {
      setState({ kind: "error", message: "Upload a PDF or Word (.docx) file." });
      return;
    }
    setElapsed(0);
    setState({ kind: "working", name: file.name, started: Date.now() });
    const body = new FormData();
    body.append("file", file);
    body.append("correction", String(correction));
    body.append("allow_unusual", String(allowUnusual));
    try {
      const res = await fetch("/api/analyze", { method: "POST", body });
      if (!res.ok) {
        const { message, soft } = await readError(res);
        setState({ kind: "error", message, soft, file });
        return;
      }
      setState({ kind: "done", report: await res.json(), name: file.name });
    } catch {
      setState({ kind: "error", message: "Couldn't reach the analysis service. Is the API running on port 8000?" });
    }
  }

  const working = state.kind === "working";
  const disabled = working || blocked;

  return (
    <>
      <p className="tag">Resume check</p>
      <h1 className="serif">Check your resume against real job postings</h1>
      <p className="lede">
        Upload a resume. ResumeMatch finds the closest postings among 107,528 from LinkedIn, checks each requirement against
        what your resume actually says, and shows you what to work on before you apply.
      </p>

      {problems.map((p) => (
        <p key={p.code} className={p.blocking ? "error" : "small muted"} style={{ marginTop: 16 }}>{p.message}</p>
      ))}

      <div
        className={`drop${over ? " over" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          const file = e.dataTransfer.files[0];
          if (file && !disabled) analyze(file);
        }}
      >
        <div>
          <p style={{ fontWeight: 500 }}>Drop your resume here</p>
          <p className="small muted">PDF or Word, up to 5 MB</p>
        </div>
        <button className="button" disabled={disabled} onClick={() => input.current?.click()}>
          Choose a file
        </button>
        <input
          ref={input} type="file" accept=".pdf,.docx" hidden
          onChange={(e) => { const f = e.target.files?.[0]; if (f) analyze(f); e.target.value = ""; }}
        />
      </div>
      <label className="options">
        <input type="checkbox" checked={correction} onChange={(e) => setCorrection(e.target.checked)} disabled={disabled} />
        Fix typos before searching (your resume is still checked as written)
      </label>
      <p className="small muted" style={{ marginTop: 6 }}>
        Your resume is processed by a local model and deleted as soon as the check finishes.
      </p>

      {working && (
        <div className="working" role="status" aria-live="polite">
          Checking <b style={{ fontWeight: 500 }}>{state.name}</b> against similar postings.
          <span className="muted"> {elapsed} s, usually about 20.</span>
          <div className="bar"><div /></div>
        </div>
      )}
      {state.kind === "error" && (
        <div className="error-box" role="alert">
          <p className="error">{state.message}</p>
          {state.soft && state.file && (
            <button className="button ghost" onClick={() => analyze(state.file!, true)}>Check it anyway</button>
          )}
        </div>
      )}
      {state.kind === "done" && <ReportView report={state.report} fileName={state.name} />}
    </>
  );
}
