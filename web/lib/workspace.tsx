"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { Report } from "@/lib/types";
import { appendFilters, type FilterValues } from "@/components/Filters";
import { handleSignedOut } from "@/lib/session";

// The report and everything made from it, shared by every page and the
// assistant. Kept in sessionStorage: it survives moving between pages and a
// reload, and is gone when the tab closes. Nothing is stored on the server.

export type TailorKind = "bullets" | "letter" | "interview";
type TailorResults = Record<string, Partial<Record<TailorKind, unknown>>>;   // job_id -> kind -> response

type Analysis = { name: string; started: number } | null;
type Failure = { message: string; soft: boolean; file?: File } | null;

interface Workspace {
  report: Report | null;
  fileName: string | null;
  analyzing: Analysis;
  failure: Failure;
  tailored: TailorResults;
  tailoring: { job_id: string; kind: TailorKind } | null;
  analyze: (file: File, opts: { correction: boolean; filters: FilterValues; allowUnusual?: boolean }) => Promise<boolean>;
  runTailor: (jobId: string, kind: TailorKind) => Promise<{ ok: boolean; message?: string }>;
  clear: () => void;
}

const KEY = "rm_workspace_v1";
const Ctx = createContext<Workspace | null>(null);

// The API answers errors as {detail: {code, message, soft}}; anything else gets a plain fallback.
export async function readError(res: Response): Promise<{ message: string; soft: boolean }> {
  const fallback = "Something went wrong on the server. Try again in a moment.";
  try {
    const detail = (await res.json()).detail;
    if (detail && typeof detail === "object") return { message: detail.message || fallback, soft: !!detail.soft };
    return { message: typeof detail === "string" ? detail : fallback, soft: false };
  } catch {
    return { message: fallback, soft: false };
  }
}

function load(): { report: Report | null; fileName: string | null; tailored: TailorResults } {
  try {
    const raw = sessionStorage.getItem(KEY);
    if (raw) return JSON.parse(raw);
  } catch { /* private mode or quota: start empty */ }
  return { report: null, fileName: null, tailored: {} };
}

export function WorkspaceProvider({ children }: { children: React.ReactNode }) {
  const [report, setReport] = useState<Report | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [tailored, setTailored] = useState<TailorResults>({});
  const [analyzing, setAnalyzing] = useState<Analysis>(null);
  const [failure, setFailure] = useState<Failure>(null);
  const [tailoring, setTailoring] = useState<{ job_id: string; kind: TailorKind } | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    const saved = load();
    setReport(saved.report); setFileName(saved.fileName); setTailored(saved.tailored || {});
    setLoaded(true);
  }, []);

  useEffect(() => {
    if (!loaded) return;
    try { sessionStorage.setItem(KEY, JSON.stringify({ report, fileName, tailored })); } catch { /* ignore */ }
  }, [report, fileName, tailored, loaded]);

  const analyze: Workspace["analyze"] = useCallback(async (file, { correction, filters, allowUnusual = false }) => {
    if (![".pdf", ".docx"].some((ext) => file.name.toLowerCase().endsWith(ext))) {
      setFailure({ message: "Upload a PDF or Word (.docx) file.", soft: false });
      return false;
    }
    setFailure(null);
    setAnalyzing({ name: file.name, started: Date.now() });
    const body = new FormData();
    body.append("file", file);
    body.append("correction", String(correction));
    body.append("allow_unusual", String(allowUnusual));
    appendFilters(body, filters);
    try {
      const res = await fetch("/api/analyze", { method: "POST", body });
      if (handleSignedOut(res)) return false;
      if (!res.ok) {
        const { message, soft } = await readError(res);
        setFailure({ message, soft, file });
        return false;
      }
      setReport(await res.json());
      setFileName(file.name);
      setTailored({});
      return true;
    } catch {
      setFailure({ message: "Couldn't reach the analysis service. Is the API running on port 8000?", soft: false });
      return false;
    } finally {
      setAnalyzing(null);
    }
  }, []);

  const runTailor: Workspace["runTailor"] = useCallback(async (jobId, kind) => {
    const job = report?.jobs.find((j) => j.job_id === jobId);
    if (!report || !job) return { ok: false, message: "That posting isn't in your report." };
    setTailoring({ job_id: jobId, kind });
    try {
      const res = kind === "interview"
        ? await fetch("/api/interview", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              job_id: jobId, resume_text: report.resume_text,
              requirements: job.requirements.map((r) => ({ text: r.text, resume_quote: r.met ? r.resume_quote : null, credential: r.credential })),
            }),
          })
        : await fetch("/api/tailor", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              kind, job_id: jobId, resume_text: report.resume_text,
              evidence: job.requirements.filter((r) => r.met && r.resume_quote).map((r) => ({ requirement: r.text, resume_quote: r.resume_quote })),
            }),
          });
      if (handleSignedOut(res)) return { ok: false };
      if (!res.ok) return { ok: false, message: (await readError(res)).message };
      const data = await res.json();
      setTailored((t) => ({ ...t, [jobId]: { ...(t[jobId] || {}), [kind]: data } }));
      return { ok: true };
    } catch {
      return { ok: false, message: "Couldn't reach the analysis service. Is the API running on port 8000?" };
    } finally {
      setTailoring(null);
    }
  }, [report]);

  const clear = useCallback(() => {
    setReport(null); setFileName(null); setTailored({}); setFailure(null);
  }, []);

  return (
    <Ctx.Provider value={{ report, fileName, analyzing, failure, tailored, tailoring, analyze, runTailor, clear }}>
      {children}
    </Ctx.Provider>
  );
}

export function useWorkspace(): Workspace {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useWorkspace outside WorkspaceProvider");
  return ctx;
}

export function jobLabel(job: { title: string; company: string | null }): string {
  return job.company ? `${job.title} at ${job.company}` : job.title;
}
