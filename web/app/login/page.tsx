"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import * as I from "@/components/Icons";
import { homeFor, type Role } from "@/lib/session";

type DemoAccount = { email: string; role_label: string; does: string };

function LoginForm() {
  const next = useSearchParams().get("next");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [working, setWorking] = useState(false);
  const [accounts, setAccounts] = useState<DemoAccount[]>([]);
  const [demoPassword, setDemoPassword] = useState<string | null>(null);
  const submitButton = useRef<HTMLButtonElement>(null);
  const [postings, setPostings] = useState<number | null>(null);

  useEffect(() => {
    fetch("/api/health").then((r) => (r.ok ? r.json() : null)).then((h) => setPostings(h?.postings ?? null)).catch(() => {});
  }, []);

  // The API lists the demo accounts, and the shared password when the
  // operator has made it public (DEMO_PASSWORD_PUBLIC=1).
  useEffect(() => {
    fetch("/api/auth/demo-accounts")
      .then((r) => (r.ok ? r.json() : null))
      .then((b) => { if (b) { setAccounts(b.accounts); setDemoPassword(b.password); } })
      .catch(() => {});
  }, []);

  function pick(account: DemoAccount) {
    setEmail(account.email);
    if (demoPassword) setPassword(demoPassword);
    setError(null);
    submitButton.current?.focus();
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setWorking(true); setError(null);
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, password }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) { setError(body?.detail?.message || "Sign-in failed. Try again."); return; }
      // Only same-site paths are followed after sign-in.
      const target = next && next.startsWith("/") && !next.startsWith("//") ? next : homeFor(body.user.role as Role);
      window.location.href = target;
    } catch {
      setError("Couldn't reach the analysis service. Is the API running on port 8000?");
    } finally {
      setWorking(false);
    }
  }

  return (
    <div className="auth-card">
      <section className="auth-side">
        <div className="brand"><span className="brand-mark" aria-hidden="true">R</span><span>ResumeMatch</span></div>
        <h1>Know which real jobs your resume fits, before you apply.</h1>
        <p>Your resume against {postings ? `${postings.toLocaleString()} ` : ""}real LinkedIn postings, with every claim checked against what you wrote.</p>
        <ul className="auth-points">
          <li><I.Check size={16} /> The closest postings and each requirement, with the line of your resume that shows it</li>
          <li><I.Check size={16} /> Resume edits, a cover letter and interview prep for any posting, with invented facts removed</li>
          <li><I.Check size={16} /> An assistant that answers from your resume and the postings, with sources</li>
          <li><I.Check size={16} /> For recruiters: rank candidates for a job, with blind review</li>
        </ul>
      </section>

      <section className="auth-form">
        <p className="eyebrow">Sign in</p>
        <h2>Welcome</h2>
        <p className="muted small" style={{ margin: 0 }}>
          This is a demo with one account per role.{" "}
          {demoPassword ? "Pick a role to fill in its email and password." : "All four share the demo password set on the server."}
        </p>
        <form className="role-form" onSubmit={submit}>
          <label>
            Email
            <input type="text" inputMode="email" autoComplete="username" value={email} onChange={(e) => setEmail(e.target.value)} required />
          </label>
          <label>
            Password
            <input type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} required />
          </label>
          <div className="row" style={{ marginTop: 0 }}>
            <button ref={submitButton} className="btn primary" type="submit" disabled={working}>{working ? "Signing in" : "Sign in"}</button>
          </div>
        </form>
        {error && <div className="error-box" role="alert"><p className="error">{error}</p></div>}

        {accounts.length > 0 && (
          <div className="demo-list" aria-label="Demo accounts">
            {accounts.map((a) => (
              <div className={`demo-item${email === a.email ? " on" : ""}`} key={a.email}>
                <span><b>{a.role_label}</b>{a.does}</span>
                <button type="button" className="btn ghost small" aria-pressed={email === a.email} onClick={() => pick(a)}
                        aria-label={`Fill in the ${a.role_label} account`}>
                  {demoPassword ? "Fill in" : "Use email"}
                </button>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

export default function LoginPage() {
  return <Suspense><LoginForm /></Suspense>;
}
